# Catalog — `aerisplane.catalog`

Hardware database of real-world components for direct use in `PropulsionSystem`
and `Aircraft` definitions.

---

## Browsing the catalog

```python
from aerisplane.catalog import list_motors, list_batteries, list_propellers, list_servos

motors = list_motors()
for m in sorted(motors, key=lambda x: x.kv):
    print(f"{m.name:40s}  KV={m.kv:5.0f}  I_max={m.max_current:.0f}A  mass={m.mass*1000:.0f}g")
```

---

## Using catalog items

```python
from aerisplane.catalog.motors    import sunnysky_x2216_1250
from aerisplane.catalog.batteries import tattu_4s_5200
from aerisplane.catalog.propellers import apc_10x4_7sf
from aerisplane.core.propulsion import ESC, PropulsionSystem

propulsion = PropulsionSystem(
    motor=sunnysky_x2216_1250,
    propeller=apc_10x4_7sf,
    battery=tattu_4s_5200,
    esc=ESC(name="generic_60A", max_current=60.0, mass=0.025),
)
```

---

## Motors

Import from `aerisplane.catalog.motors`.

| Variable | Name | KV | Max I [A] | Mass [g] |
|---|---|---|---|---|
| `sunnysky_x2216_1250` | SunnySky X2216 1250KV | 1250 | 28 | 58 |
| `sunnysky_x2216_2400` | SunnySky X2216 2400KV | 2400 | 28 | 58 |
| `sunnysky_x2212_980` | SunnySky X2212 980KV | 980 | 20 | 52 |
| `tiger_mn3110_700` | T-Motor MN3110 700KV | 700 | 16 | 102 |
| `tiger_mn3110_780` | T-Motor MN3110 780KV | 780 | 16 | 102 |
| `tiger_mn2213_950` | T-Motor MN2213 950KV | 950 | 14 | 60 |
| `tiger_mn4014_330` | T-Motor MN4014 330KV | 330 | 22 | 176 |
| `tiger_mn5212_340` | T-Motor MN5212 340KV | 340 | 30 | 215 |
| `t_motor_f80_1900` | T-Motor F80 1900KV | 1900 | 30 | 68 |
| `t_motor_f60_2550` | T-Motor F60 2550KV | 2550 | 35 | 55 |
| `emax_mt2213_935` | Emax MT2213 935KV | 935 | 20 | 57 |
| `emax_mt2216_810` | Emax MT2216 810KV | 810 | 20 | 75 |
| `emax_rs2205_2600` | Emax RS2205 2600KV | 2600 | 30 | 30 |
| `rctimer_5010_360` | RCTimer 5010 360KV | 360 | 40 | 190 |
| `scorpion_m2205_2350` | Scorpion M2205 2350KV | 2350 | 30 | 35 |
| `scorpion_hkii_2221_900` | Scorpion HKII-2221 900KV | 900 | 22 | 68 |
| `axi_2217_20` | AXi 2217/20 | 1050 | 18 | 95 |
| `turnigy_d3530_1400` | Turnigy D3530/14 1400KV | 1400 | 21 | 86 |
| `hacker_a20_26` | Hacker A20-26L | 1020 | 16 | 72 |
| `dualsky_eco_2315c_1100` | Dualsky ECO 2315C 1100KV | 1100 | 18 | 65 |

---

## Batteries

Import from `aerisplane.catalog.batteries`.

| Variable | Name | Cap [Ah] | Voltage [V] | C-rating | Mass [g] |
|---|---|---|---|---|---|
| `tattu_3s_2300` | Tattu 3S 2300mAh 45C | 2.3 | 11.1 | 45 | 178 |
| `tattu_4s_1800` | Tattu 4S 1800mAh 75C | 1.8 | 14.8 | 75 | 218 |
| `tattu_4s_3300` | Tattu 4S 3300mAh 45C | 3.3 | 14.8 | 45 | 302 |
| `tattu_4s_5200` | Tattu 4S 5200mAh 45C | 5.2 | 14.8 | 45 | 470 |
| `tattu_6s_10000` | Tattu 6S 10000mAh 25C | 10.0 | 22.2 | 25 | 1280 |
| `tattu_6s_16000` | Tattu 6S 16000mAh 15C | 16.0 | 22.2 | 15 | 1900 |
| `gens_ace_3s_2200` | Gens Ace 3S 2200mAh 25C | 2.2 | 11.1 | 25 | 162 |
| `gens_ace_4s_4000` | Gens Ace 4S 4000mAh 45C | 4.0 | 14.8 | 45 | 390 |
| `gens_ace_6s_6000` | Gens Ace 6S 6000mAh 30C | 6.0 | 22.2 | 30 | 870 |
| `turnigy_nano_tech_3s_2200` | Turnigy Nano-tech 3S 2200mAh | 2.2 | 11.1 | 25 | 156 |
| `turnigy_nano_tech_4s_5000` | Turnigy Nano-tech 4S 5000mAh | 5.0 | 14.8 | 25 | 480 |
| `turnigy_nano_tech_6s_3300` | Turnigy Nano-tech 6S 3300mAh | 3.3 | 22.2 | 45 | 480 |
| `multistar_4s_10000` | Multistar 4S 10000mAh 10C | 10.0 | 14.8 | 10 | 890 |
| `ovonic_4s_2200` | Ovonic 4S 2200mAh 50C | 2.2 | 14.8 | 50 | 200 |
| `ovonic_6s_3300` | Ovonic 6S 3300mAh 50C | 3.3 | 22.2 | 50 | 480 |

---

## Propellers

Import from `aerisplane.catalog.propellers`.

| Variable | Diameter × Pitch | Mass [g] |
|---|---|---|
| `apc_10x4_7sf` | 10×4.7 in | 18 |
| `apc_11x4_7sf` | 11×4.7 in | 21 |
| `apc_13x4_7sf` | 13×4.7 in | 30 |
| `apc_10x7e` | 10×7 in | 19 |
| `apc_12x6e` | 12×6 in | 27 |
| `apc_14x8_3mf` | 14×8.3 in | 40 |
| `master_airscrew_10x5` | 10×5 in | 20 |
| `master_airscrew_11x7` | 11×7 in | 26 |
| `master_airscrew_14x7` | 14×7 in | 42 |
| `tjd_14x8_5` | 14×8.5 in | 22 |

---

## Airfoils

The airfoil catalog contains 2175 `.dat` files. Load by name:

```python
import aerisplane as ap
af = ap.Airfoil(name="ag35")       # load from catalog
af = ap.Airfoil(name="naca2412")   # generated analytically (NACA 4-digit)
```

---

## Auto-generated reference

::: aerisplane.catalog.list_motors

::: aerisplane.catalog.list_batteries

::: aerisplane.catalog.list_propellers

::: aerisplane.catalog.list_servos

::: aerisplane.catalog.get_airfoil
