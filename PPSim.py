import pandapower as pp

# Base values (feel free to change
S_BASE = 100.0
VN_HV, VN_LV = 110.0, 20.0

net = pp.create_empty_network(name="IEEE 14-bus", sn_mva=S_BASE, f_hz=60.0)

# These are the indexes that connect all loads, generators, high V, low V, and transformers together.
HV_LINES = [(1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)]
LV_LINES = [(6, 11), (6, 12), (6, 13), (7, 8), (7, 9), (9 ,10), (9, 14), (10, 11), (12, 13), (13, 14)]
GENS_IDX = [(2, 40.0, 1.045), (3, 0.0, 1.010), (6, 0.0, 1.070), (8, 0.0, 1.090)]
TR_LINES = [(4, 7), (4, 9), (5, 6)]
LOAD_IDX = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]

# These are the populated lists (for living life secure).
bus = {}
load = {}
lv_lines = {}
hv_lines = {}
tr_lines = {}

# Here, we create the buses, from bus 1-15, where 1-5 are HV
for i in range(1, 15):
    vn = VN_HV if i <= 5 else VN_LV
    bus[i] = pp.create_bus(net, vn_kv=vn, name=f"Bus{i}")

# Here, we create the loads, which exist at certain buses.
for j in LOAD_IDX:
    load[j] = pp.create_load(net, bus=bus[j], p_mw=2, name=f"Load{j}")

# Here, we create the lines. We start with the high voltage lines, low voltage lines, and then the transformers.
for d, b in HV_LINES:
    hv_lines[(d, b)] = pp.create_line(net, bus[d], bus[b], 12, "149-AL1/24-ST1A 110.0", f"line{d}-{b}")
for m, n in LV_LINES:
    lv_lines[(m, n)] = pp.create_line(net, bus[m], bus[n], 12, "149-AL1/24-ST1A 20.0", f"line{m}-{n}")
for q, p in TR_LINES:
    tr_lines[(q, p)] = pp.create_transformer(net, bus[q], bus[p], "63 MVA 110/20 kV", f"transform{q}-{p}")

# Since bus one has the slack, we create the ext_grid at bus 1, then the rest at their respective index.
pp.create_ext_grid(net, bus[1], vm_pu=1.02, name="Slack")
for b, p, v in GENS_IDX:
    pp.create_gen(net, bus[b], p, v, name=f"Gen{b}")

# This is the run. We compute, then print all of the silly nice values for viewing purposes :3
pp.runpp(net)
print(net.res_bus)
print(net.res_line)
print(net.res_trafo)
print(net.res_gen)
print(net.res_ext_grid)