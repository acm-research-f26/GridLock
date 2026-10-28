"""Self-contained GAT autoencoder demo. Requires only PyTorch.
Synthetic 39-node signals and a toy graph, NOT IEEE39 power-flow data.
Run: python gat_all_in_one.py --epochs 2
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.0):
        super().__init__()
        self.heads, self.width, self.concat = heads, out_features, concat
        self.lin_src = nn.Linear(in_features, heads*out_features, bias=False)
        self.lin_tgt = nn.Linear(in_features, heads*out_features, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, out_features))
        self.bias = nn.Parameter(torch.zeros(heads*out_features if concat else out_features))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_tgt.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edges):
        n = x.size(0)
        source, target = edges
        src = self.lin_src(x).reshape(n, self.heads, self.width)
        tgt = self.lin_tgt(x).reshape(n, self.heads, self.width)
        scores = (F.leaky_relu(src[source]+tgt[target], 0.2)*self.att).sum(-1)
        index = target[:, None].expand(-1, self.heads)
        maximum = scores.new_full((n, self.heads), -torch.inf)
        maximum.scatter_reduce_(0, index, scores.detach(), reduce="amax", include_self=True)
        weights = (scores-maximum[target]).exp()
        denominator = scores.new_zeros((n, self.heads))
        denominator.scatter_add_(0, index, weights)
        alpha = weights/denominator[target].clamp_min(torch.finfo(weights.dtype).tiny)
        messages = self.dropout(alpha).unsqueeze(-1)*src[source]
        result = messages.new_zeros((n, self.heads, self.width))
        result.index_add_(0, target, messages)
        result = result.reshape(n, -1) if self.concat else result.mean(1)
        return result+self.bias


class GATAutoencoder(nn.Module):
    def __init__(self, n_features=4, hidden=16, heads=4, latent=32, dropout=0.0):
        super().__init__()
        self.gat1 = GATv2Layer(n_features, hidden, heads, True, dropout)
        self.gat2 = GATv2Layer(heads*hidden, hidden, heads, False, dropout)
        self.enc_lstm = nn.LSTM(hidden, latent, batch_first=True)
        self.dec_lstm = nn.LSTM(latent, latent, batch_first=True)
        self.out = nn.Linear(latent, n_features)
        self.register_buffer("threshold", torch.tensor(float("nan")))

    def forward(self, x, edge_index):
        b, t, n, features = x.shape
        offsets = torch.arange(b*t, device=x.device).reshape(-1, 1, 1)*n
        edges = (edge_index[None]+offsets).permute(1,0,2).reshape(2,-1)
        h = F.elu(self.gat1(x.reshape(-1, features), edges))
        h = F.elu(self.gat2(h, edges))
        h = h.reshape(b,t,n,-1).permute(0,2,1,3).reshape(b*n,t,-1)
        _, (hidden, _) = self.enc_lstm(h)
        z = hidden[-1].reshape(b,n,-1)
        repeated = z.reshape(b*n,1,-1).expand(-1,t,-1).contiguous()
        decoded, _ = self.dec_lstm(repeated)
        reconstruction = self.out(decoded).reshape(b,n,t,features).permute(0,2,1,3)
        return reconstruction, z


def make_data(nodes=39, features=4, steps=2880, window=24, stride=4, seed=42):
    generator = torch.Generator().manual_seed(seed)
    phase = torch.rand(nodes,features,generator=generator)*0.6-0.3
    amplitude = 0.7+0.6*torch.rand(nodes,features,generator=generator)
    time = torch.arange(steps).reshape(-1,1,1)
    noise = torch.zeros(steps,nodes,features)
    innovations = torch.randn(steps,nodes,features,generator=generator)*0.04
    for i in range(1,steps):
        noise[i] = 0.9*noise[i-1]+innovations[i]
    raw = (amplitude*torch.sin(2*torch.pi*time/96+phase)
           +0.25*torch.cos(2*torch.pi*time/672+phase)+noise)
    cut1, cut2 = int(steps*0.7), int(steps*0.85)
    if min(cut1,cut2-cut1,steps-cut2) < window:
        raise ValueError("Each split must contain a complete window; increase --steps")
    mean = raw[:cut1].mean(0)
    scale = raw[:cut1].std(0,correction=0).clamp_min(1e-6)
    values = (raw-mean)/scale
    splits, ends = {}, {}
    for name, start, stop in (("train",0,cut1),("validation",cut1,cut2),("test",cut2,steps)):
        indices = list(range(start+window-1,stop,stride))
        splits[name] = torch.stack([values[i-window+1:i+1] for i in indices])
        ends[name] = torch.tensor(indices)
    pairs = {(i,i) for i in range(nodes)}
    for i in range(nodes):
        for offset in (1,5):
            j = (i+offset)%nodes
            pairs.update(((i,j),(j,i)))
    edges = torch.tensor(sorted(pairs),dtype=torch.long).T.contiguous()
    return splits, edges, mean, scale, ends


@torch.no_grad()
def score(model, data, edges, batch_size, device):
    previous = [(m,m.training) for m in model.modules()]
    results = []
    try:
        model.eval()
        for batch in data.split(batch_size):
            batch = batch.to(device)
            reconstruction, _ = model(batch,edges)
            errors = (reconstruction-batch).square().mean((1,3))
            if not torch.isfinite(errors).all():
                raise RuntimeError("Nonfinite reconstruction scores")
            results.append(errors.cpu())
    finally:
        for module, mode in previous:
            module.training = mode
    buses = torch.cat(results)
    return buses.mean(1), buses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--steps",type=int,default=2880)
    parser.add_argument("--nodes",type=int,default=39)
    parser.add_argument("--features",type=int,default=4)
    parser.add_argument("--window",type=int,default=24)
    parser.add_argument("--stride",type=int,default=4)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--quantile",type=float,default=0.99)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--save",default="gat_model.pt")
    args = parser.parse_args()
    for name in ("epochs","steps","nodes","features","window","stride","batch_size","threads"):
        if getattr(args,name) < 1:
            parser.error(f"{name} must be positive")
    if not 0 < args.quantile < 1 or not 0 < args.lr < float("inf"):
        parser.error("Invalid quantile or learning rate")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    splits, edges, mean, scale, ends = make_data(
        args.nodes,args.features,args.steps,args.window,args.stride,args.seed)
    print("Synthetic toy-graph data: NOT IEEE39 power-flow measurements")
    for name, data in splits.items():
        print(name,tuple(data.shape))
    hp = dict(n_features=args.features,hidden=16,heads=4,latent=32,dropout=0.0)
    model = GATAutoencoder(**hp).to(device)
    graph = edges.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    loader = DataLoader(TensorDataset(splits["train"]),batch_size=args.batch_size,shuffle=True)
    best_loss, best_state = float("inf"), None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch,graph)
            loss = F.mse_loss(reconstruction,batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total += loss.item()*len(batch)
        val, _ = score(model,splits["validation"],graph,args.batch_size,device)
        val_loss = val.mean().item()
        train_loss = total/len(splits["train"])
        history.append({"epoch":epoch+1,"train_loss":train_loss,"val_loss":val_loss})
        print(f"Epoch {epoch+1}/{args.epochs}: train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    validation, _ = score(model,splits["validation"],graph,args.batch_size,device)
    threshold = torch.quantile(validation,args.quantile).item()
    with torch.no_grad():
        model.threshold.fill_(threshold)
    test_scores, bus_scores = score(model,splits["test"],graph,args.batch_size,device)
    flags = test_scores > threshold
    print(f"Threshold: {threshold:.6f}; flagged normal test windows: {int(flags.sum())}/{len(flags)}")
    print("This flag rate is not attack-detection accuracy.")
    if args.save:
        destination = Path(args.save)
        destination.parent.mkdir(parents=True,exist_ok=True)
        checkpoint = {
            "model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
            "hyperparameters":hp,"threshold":threshold,"edge_index":edges,
            "mean":mean,"scale":scale,"window":args.window,"node_count":args.nodes,
            "seed":args.seed,"history":history,"test_scores":test_scores,
            "test_bus_scores":bus_scores,"test_flags":flags,
            "test_end_timesteps":ends["test"],
            "description":"Synthetic correlated signals on a toy graph; not IEEE39"
        }
        torch.save(checkpoint,destination)
        print(f"Saved checkpoint and test results: {destination.resolve()}")


if __name__ == "__main__":
    main()
