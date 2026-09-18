# OKAY SO BASICALLY, WE NEED TO BALANCE THE VOLTAGE AND DISTANCE! OUR LOSSES ARE CURRENTLY WAY TOO GOOD,
# AND SO WE NEED TO BRING OUT DOWN BY INCREASING DISTANCE TO INCREASE LOSS, AND THEN INCREASING/DECREASING VOLTAGE.

import pandapower as pp
import pandas as pd

# Base values (feel free to change).
S_BASE = 100.0
VN_HV, VN_LV = 110.0, 20.0

net = pp.create_empty_network(name="IEEE 14-bus", sn_mva=S_BASE, f_hz=60.0)

# These are the indexes that connect all loads, generators, high V, low V, and transformers together.
HV_LINES = [(1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)]
LV_LINES = [(6, 11), (6, 12), (6, 13), (7, 8), (7, 9), (9 ,10), (9, 14), (10, 11), (12, 13), (13, 14)]
GENS_IDX = [(2, 40.0, 1.045, -40, 50), (3, 0.0, 1.010, 0, 40), (6, 0.0, 1.070, -6, 24), (8, 0.0, 1.090, -6, 24)]
TR_LINES = [(4, 7), (4, 9), (5, 6)]
LOAD_IDX = [(2, 21.7, 12.7), (3, 94.2, 19), (4, 47.8, -3.9), (5, 7.6, 1.6), (6, 11.2, 7.5), (9, 29.5, 16.6),
            (10, 9.0, 5.8), (11, 3.5, 1.8), (12, 6.1, 1.6), (13, 13.5, 5.8), (14, 14.9, 5.0)]

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
for j, pm, qm in LOAD_IDX:
    load[j] = pp.create_load(net, bus=bus[j], p_mw=pm, q_mvar= qm, name=f"Load{j}")

# Here, we create the lines. We start with the high voltage lines, low voltage lines, and then the transformers.
for d, b in HV_LINES:
    hv_lines[(d, b)] = pp.create_line(net, bus[d], bus[b], 2, "243-AL1/39-ST1A 110.0", f"line{d}-{b}")
for m, n in LV_LINES:
    lv_lines[(m, n)] = pp.create_line(net, bus[m], bus[n], 2, "243-AL1/39-ST1A 20.0", f"line{m}-{n}")
for q, p in TR_LINES:
    tr_lines[(q, p)] = pp.create_transformer(net, bus[q], bus[p], "63 MVA 110/20 kV", f"transform{q}-{p}")

# Since bus one has the slack, we create the ext_grid at bus 1, then the rest at their respective index.
pp.create_ext_grid(net, bus[1], vm_pu=1.06, name="Slack")
pp.create_shunt(net, bus[9], q_mvar=-19.0)

for b, p, v, q_min, q_max in GENS_IDX:
    pp.create_gen(net, bus[b], p, v, min_q_mvar=q_min, max_q_mvar=q_max, name=f"Gen{b}")

net.trafo["shift_degree"] = 0

# This is the run. We compute, then print all the silly nice values for viewing purposes :3
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

pp.runpp(net, enforce_q_lims=True)
t = net.res_line.assign(name=net.line.name)[["name", "loading_percent"]]
print(t[t.loading_percent > 80].sort_values("loading_percent", ascending=False).round(1).to_string(index=False) + '\n')

b = net.res_bus.assign(name=net.bus.name)[["name", "vm_pu"]]
print(b[(b.vm_pu < 0.95) | (b.vm_pu > 1.05)].sort_values("vm_pu").round(3).to_string(index=False) + '\n')

g = net.res_gen.assign(name=net.gen.name)[["name", "p_mw", "q_mvar", "vm_pu"]]
print(g.round(1).to_string(index=False) + '\n')

print(net.res_trafo.assign(name=net.trafo.name)[["name", "loading_percent"]].round(1).to_string(index=False) + '\n')

print(f"losses: {net.res_line.pl_mw.sum():.1f} MW")
print(f"slack:  {net.res_ext_grid.p_mw.iloc[0]:.1f} MW, {net.res_ext_grid.q_mvar.iloc[0]:.1f} MVAr")
