# RC Autonomous Platform — Design Requirements & Goals

**Project:** Modular fixed-wing RC aircraft for autonomous systems research  
**Configuration:** Mid-wing · Modular wing + tail · 3D print + carbon composite  
**Autonomy stack:** ArduPlane / PX4 · MAVLink · Companion computer ready  
**Version:** 0.1 (draft)  
**Date:** 2026-03-16

---

## Table of Contents

1. [Mission & Platform Goals](#1-mission--platform-goals)
2. [Airframe Sizing](#2-airframe-sizing)
3. [Modular Wing Interface](#3-modular-wing-interface)
4. [Modular Tail Interface](#4-modular-tail-interface)
5. [Aerodynamic Stability Requirements](#5-aerodynamic-stability-requirements)
6. [Structural Requirements](#6-structural-requirements)
7. [Propulsion System](#7-propulsion-system)
8. [Avionics & Autonomy Bay](#8-avionics--autonomy-bay)
9. [Autopilot Parameter Management](#9-autopilot-parameter-management)
10. [Electrical System](#10-electrical-system)
11. [Safety Requirements](#11-safety-requirements)
12. [Regulatory & Legal Requirements](#12-regulatory--legal-requirements)
13. [Manufacturing & Repairability](#13-manufacturing--repairability)
14. [Testing & Validation Protocol](#14-testing--validation-protocol)
15. [Documentation Requirements](#15-documentation-requirements)
16. [Commercial & Open-Source Considerations](#16-commercial--open-source-considerations)
17. [Future / Optional Goals](#17-future--optional-goals)
18. [Open Questions & Decisions Pending](#18-open-questions--decisions-pending)

---

## Priority Legend

| Tag | Meaning |
|-----|---------|
| `[MUST]` | Non-negotiable. Design fails without this. |
| `[SHOULD]` | Strong intent. Deviate only with documented justification. |
| `[NICE]` | Desirable if cost/effort allows. |
| `[OPEN]` | Decision not yet made. Needs resolution before detailed design. |

---

## 1. Mission & Platform Goals

| ID | Priority | Requirement |
|----|----------|-------------|
| M-01 | `[MUST]` | The platform shall serve as a reconfigurable autonomous systems testbed — supporting sensor integration, autopilot tuning, mission planning, and algorithm validation. |
| M-02 | `[MUST]` | The fuselage shall be permanent across all configurations. Wings and tail are the only interchangeable assemblies. |
| M-03 | `[MUST]` | All wing + tail combinations shall be flyable under ArduPlane or PX4 firmware without hardware modification to the fuselage. |
| M-04 | `[SHOULD]` | The platform shall support a companion computer (Raspberry Pi 4, Jetson Nano, or equivalent) for onboard processing tasks such as computer vision, SLAM, and path planning. |
| M-05 | `[SHOULD]` | The platform shall be usable as a repeatable research instrument — i.e. two builds from the same files shall produce aerodynamically and electrically equivalent aircraft. |
| M-06 | `[NICE]` | The platform interface standard shall be published openly to allow third-party wing and tail variants to be developed by the community. |

---

## 2. Airframe Sizing

| ID | Priority | Requirement |
|----|----------|-------------|
| S-01 | `[MUST]` | Wingspan: 1400 – 1800 mm across all wing variants. |
| S-02 | `[MUST]` | Fuselage length: 900 – 1100 mm. |
| S-03 | `[MUST]` | Maximum take-off weight (MTOW): 2.5 – 3.5 kg including all payload, battery, and avionics. |
| S-04 | `[MUST]` | Wing loading shall not exceed 35 g/dm² at MTOW for any wing variant, to maintain a safe stall speed for autonomous operation. |
| S-05 | `[SHOULD]` | Fuselage cross-section shall accommodate a minimum internal width of 120 mm and height of 100 mm for the avionics bay. |
| S-06 | `[SHOULD]` | Overall footprint when disassembled shall fit within a 1200 × 400 × 300 mm transport case. |

> **💡 Advice:** Define MTOW early and freeze it. Every downstream decision — spar diameter, motor kV, battery capacity — depends on weight budget. Maintain a live weight budget spreadsheet from day one, with a target, current estimate, and margin column for every component.

---

## 3. Modular Wing Interface

> The wing interface is the most critical design decision in this project. Once the fuselage is built, this geometry is frozen.

| ID | Priority | Requirement |
|----|----------|-------------|
| W-01 | `[MUST]` | All wing variants shall use a standardised carbon spar tube interface. Spar tube outer diameter and material shall be defined and frozen before fuselage design begins. Recommended: 20 mm round or 15 × 15 mm square CF tube. |
| W-02 | `[MUST]` | Spar pocket position (distance from fuselage centreline, vertical offset) shall be identical across all wing variants. |
| W-03 | `[MUST]` | Each wing panel shall be retained by 2× M4 bolts into captive stainless steel inserts in the fuselage wing saddle. |
| W-04 | `[MUST]` | Wing root alignment shall be achieved by a minimum of 2 anti-rotation dowels (carbon or aluminium) per wing panel, separate from the spar. |
| W-05 | `[MUST]` | A standardised electrical connector shall be located at the wing root. Pinout shall include: servo power (+5 V), servo signal (aileron minimum; flap optional), and ground. Connector family shall be defined and frozen with the fuselage. |
| W-06 | `[MUST]` | Target field swap time: ≤ 10 minutes per wing pair using only a hex driver. No adhesives, no soldering. |
| W-07 | `[MUST]` | The centre of gravity (CG) for every wing variant shall fall within the battery slide range of the fuselage (see S-08). |
| W-08 | `[SHOULD]` | Wing root shall have a defined aerodynamic seal (foam gasket or 3D-printed lip) to prevent airflow separation at the fuselage-wing junction. |
| W-09 | `[SHOULD]` | Each wing variant shall ship with a datasheet specifying: span, area, root chord, tip chord, MAC, dihedral, sweep, airfoil designation, and aerodynamic centre location. |
| W-10 | `[NICE]` | Wing root connector shall include spare pins for future use (e.g. winglet sensors, heated leading edge, lighting). |

> **💡 Advice:** Prototype the wing attachment interface in cheap PLA before committing to the final fuselage material. Test spar pull-out force, twist resistance under simulated aileron load, and fit tolerance across at least 3 printed copies to validate print repeatability.

---

## 4. Modular Tail Interface

> Changing the wing changes wing area and aerodynamic centre. The tail must scale accordingly or stability is lost. Wing and tail are a **matched pair** — never swap one without re-validating the other.

| ID | Priority | Requirement |
|----|----------|-------------|
| T-01 | `[MUST]` | The tail assembly (horizontal stabiliser + vertical fin as a single removable unit) shall attach to the fuselage via a standardised carbon boom socket. Boom outer diameter shall be frozen with the fuselage design. Recommended: 20 mm OD × 2 mm wall CF tube. |
| T-02 | `[MUST]` | Tail boom shall be retained by 2× M4 bolts + 1 anti-rotation pin into the fuselage rear bulkhead. |
| T-03 | `[MUST]` | Every wing variant shall have a paired tail specification defining: horizontal stabiliser area (Sh), vertical fin area (Sv), and tail arm length (Lh, Lv). These values shall be calculated to meet the volume coefficient targets in Section 5. |
| T-04 | `[MUST]` | Mixing unpaired wing and tail combinations is prohibited without full recalculation of V̄h and V̄v and sign-off on the pre-flight checklist. |
| T-05 | `[MUST]` | Tail electrical interface shall use the same connector family as the wing root. Pinout: elevator servo, rudder servo, power, ground. No tail-specific wiring changes between swaps. |
| T-06 | `[MUST]` | Target field swap time: ≤ 5 minutes for tail assembly. Total reconfiguration (wing + tail): ≤ 15 minutes. |
| T-07 | `[SHOULD]` | Tail boom length shall be selectable from a discrete set (e.g. 400 mm, 500 mm, 600 mm) using the same socket, to allow tail arm adjustment across very different wing configurations. |
| T-08 | `[SHOULD]` | Horizontal stabiliser shall be an all-moving stabilator, or use a separate elevator with ≥ 30% chord ratio. All-moving preferred for wider trim authority during autopilot tuning. |
| T-09 | `[SHOULD]` | Rudder chord ratio shall be ≥ 25% of vertical fin chord. |
| T-10 | `[NICE]` | A V-tail variant shall be considered as a future option. V-tail angle and area shall be calculated to deliver equivalent Vh and Vv to the conventional tail it replaces. |

> **💡 Advice:** The variable boom length (T-07) is one of the most powerful tools in your modular toolkit. A longer boom allows a smaller, lighter tail to achieve the same volume coefficient. Design the boom socket to accept standard CF tube sizes from a supplier you can reliably reorder from.

---

## 5. Aerodynamic Stability Requirements

> These requirements apply to **every wing + tail combination**. They must be verified analytically before build and validated in SITL before maiden flight.

| ID | Priority | Requirement |
|----|----------|-------------|
| A-01 | `[MUST]` | Horizontal tail volume coefficient: **V̄h = 0.35 – 0.45** for all configurations. Formula: `V̄h = (Sh × Lh) / (S × c̄)` where S = wing area, c̄ = mean aerodynamic chord. |
| A-02 | `[MUST]` | Vertical tail volume coefficient: **V̄v = 0.04 – 0.06** for all configurations. Formula: `V̄v = (Sv × Lv) / (S × b)` where b = wingspan. |
| A-03 | `[MUST]` | Static longitudinal margin: **+5% to +15% MAC** for all configurations. Static margin = `(NP − CG) / MAC`. Margins below 5% risk autopilot oscillation; above 15% risk insufficient pitch authority. |
| A-04 | `[MUST]` | All wing variants shall have positive effective dihedral (roll stability, Clβ < 0). Minimum 3° geometric dihedral for straight wings. Swept wings shall calculate sweep-induced dihedral contribution. |
| A-05 | `[MUST]` | Elevator trim shall achieve level flight across the speed range 1.3 × Vstall to 2.0 × Vstall without exceeding ±15° elevator deflection (reserve authority for autopilot). |
| A-06 | `[MUST]` | Each wing/tail pair shall have a published CG envelope: forward limit (pitch authority limited) and aft limit (stability limited). Battery slide range shall span at least the midpoint ± 20 mm of that envelope. |
| A-07 | `[SHOULD]` | Aspect ratio of wing variants shall stay within 6 – 12 to remain within the aerodynamic assumptions used for volume coefficient calculations. |
| A-08 | `[SHOULD]` | Stall characteristics shall be benign: progressive, wings-level stall break, with natural recovery tendency. Tip stall or snap behaviour is not acceptable on any variant. Airfoil selection and washout shall be chosen accordingly. |
| A-09 | `[NICE]` | A simple OpenVSP or XFLR5 model shall be created for each wing variant to validate Cl/Cd, neutral point, and Cm curves before physical build. |

> **💡 Advice — airfoil selection:** For your first wing, avoid undercambered or fully symmetrical airfoils. A flat-bottomed or low-cambered section (e.g. Clark Y, NACA 2412, AG series) gives predictable stall behaviour and works well across the speed range relevant to a 3 kg autonomous platform. Symmetrical sections are for aerobatics — not the priority here.

---

## 6. Structural Requirements

| ID | Priority | Requirement |
|----|----------|-------------|
| ST-01 | `[MUST]` | Design load factor: **+6g / −3g at MTOW**. This provides margin for autopilot overshoot during tuning and unexpected gust loads. |
| ST-02 | `[MUST]` | Fuselage shall be a hybrid structure: 3D-printed shell (PETG or ASA) providing shape + aerodynamic form, with internal carbon tube/rod longerons carrying primary bending and torsion loads. A purely printed fuselage with no CF reinforcement is not acceptable. |
| ST-03 | `[MUST]` | Wing spars shall be carbon fibre tube(s). Spar sizing shall be calculated for the bending moment at the wing root at design load factor + 1.5× safety factor. |
| ST-04 | `[MUST]` | All primary control surface hinges shall be pinned CF rod or commercial carbon/nylon hinge. 3D-printed integral hinges are not acceptable for elevator, rudder, or aileron. |
| ST-05 | `[MUST]` | Tail boom attachment shall be designed as a primary structural joint. The bending moment at the boom socket at +6g / −3g load shall be the primary sizing case for the rear fuselage bulkhead. |
| ST-06 | `[MUST]` | All fasteners at structural interfaces (wing bolts, tail boom bolts) shall be stainless steel M4 minimum with thread-locking compound (Loctite 243 or equivalent). |
| ST-07 | `[SHOULD]` | Fuselage shall be designed in printable sections ≤ 250 mm in any axis to fit a standard 250 × 250 mm print bed. Sections shall join via overlapping flanges and M3 bolts, not glue alone. |
| ST-08 | `[SHOULD]` | All printed structural parts shall be printed in PETG or ASA (not PLA) to withstand temperatures inside a parked vehicle (>60°C) without deformation. |
| ST-09 | `[SHOULD]` | Wing skin options (printed ribs + film, fibreglass over foam core, or printed skin panels) shall be defined per variant. Each option shall meet the torsional stiffness requirement: wing tip twist under max aileron load ≤ 1°. |
| ST-10 | `[NICE]` | Key structural joints shall have visual inspection markings so that cracks or separation are visible during pre-flight checks without disassembly. |

> **💡 Advice — weight creep:** 3D-printed structures are heavier than they look on screen. Print a single fuselage section and weigh it before committing to the full design. Budget 20–30% weight contingency on printed parts versus CAD estimates. Use gyroid or lightning infill patterns to reduce weight; solid infill is almost never necessary.

---

## 7. Propulsion System

| ID | Priority | Requirement |
|----|----------|-------------|
| P-01 | `[MUST]` | Thrust-to-weight ratio at full throttle: ≥ 0.7 : 1. Recommended 1 : 1 to allow recovery from stalled or nose-low autopilot excursions. |
| P-02 | `[MUST]` | Battery: 4S LiPo, 4000 – 6000 mAh. Accessible via top or belly hatch. Fore/aft slideable for CG trim. |
| P-03 | `[MUST]` | Battery slide range: minimum ±25 mm from nominal CG position, to accommodate CG shifts between wing variants. |
| P-04 | `[MUST]` | Battery shall be retained by a positive locking mechanism (strap + hook, or latch). Gravity or friction retention alone is not acceptable. |
| P-05 | `[MUST]` | Endurance at 50% throttle cruise: ≥ 20 minutes. This is the minimum for a meaningful autonomous mission test cycle. |
| P-06 | `[SHOULD]` | Motor: brushless outrunner, 1000 – 1400 kV (4S), 800 – 1000 W peak. Prop size to be matched per motor and wing variant based on desired cruise speed. |
| P-07 | `[SHOULD]` | Pusher configuration preferred to leave the nose free for pitot tube, forward camera, and sensor payload. Tractor is acceptable if pusher introduces unacceptable CG or structural complexity. |
| P-08 | `[SHOULD]` | ESC shall support telemetry feedback (RPM, current, temperature) to the flight controller via BLHeli32 or AM32 with ESC telemetry protocol. |
| P-09 | `[SHOULD]` | A low-voltage cutoff shall be configured both at the ESC and in the autopilot (FS_BATT_VOLTAGE in ArduPlane) to trigger RTL before battery over-discharge. |
| P-10 | `[NICE]` | Motor and ESC mounting shall allow replacement without fuselage disassembly — accessible via a removable hatch or nose cone. |

> **💡 Advice — pusher vs tractor trade-off:** Pusher is strongly recommended for an autonomy platform. The clean nose allows a forward-facing camera, clean pitot tube airflow, and potential lidar. The main disadvantage is propeller ground strike risk on tail-low landings — design the tail skid or landing gear to prevent this. Many successful research platforms (Skywalker X8, Opterra) use pusher for exactly these reasons.

---

## 8. Avionics & Autonomy Bay

| ID | Priority | Requirement |
|----|----------|-------------|
| AV-01 | `[MUST]` | Flight controller: Pixhawk 6C, Cube Orange, or equivalent. ArduPlane or PX4 firmware. Hard-mounted to a vibration-isolating plate (rubber standoffs or foam isolation). |
| AV-02 | `[MUST]` | GPS + compass module: external, mounted on a mast ≥ 50 mm above the fuselage top to minimise magnetic interference from wiring and motors. |
| AV-03 | `[MUST]` | Airspeed sensor: pitot-static tube, mounted on the nose (pusher) or wingtip (tractor), connected to flight controller. Stall speed shall not be inferred from throttle alone. |
| AV-04 | `[MUST]` | Telemetry radio: 915 MHz (Americas) or 433 MHz (Europe/rest), SiK-based (RFD900 or equivalent). MAVLink output to ground station running Mission Planner or QGroundControl. |
| AV-05 | `[MUST]` | RC receiver connected to flight controller with hardware failsafe configured: loss of RC signal triggers Return To Launch (RTL) mode. |
| AV-06 | `[MUST]` | Manual RC override shall be possible at all times, regardless of autopilot mode. This is a non-negotiable safety requirement. |
| AV-07 | `[MUST]` | Companion computer bay: minimum 100 × 70 mm footprint reserved inside fuselage. Power (5 V / 3 A regulated) and UART or USB connection to flight controller pre-wired. |
| AV-08 | `[SHOULD]` | All avionics wiring shall be routed away from carbon fibre structural members to prevent RF interference with GPS and telemetry. Carbon fibre is conductive and attenuates RF. |
| AV-09 | `[SHOULD]` | A power distribution board or module shall provide: main battery monitoring (voltage + current), regulated 5 V BEC for avionics, and a separate regulated 5 V BEC for servos. Avionics and servo power shall not share a single BEC. |
| AV-10 | `[SHOULD]` | Payload bay: a standardised mounting plate (minimum 80 × 60 mm, 4× M3 inserts on 30 × 30 mm grid) shall be accessible at the nose or belly. |
| AV-11 | `[SHOULD]` | All connectors shall be labelled with permanent markers or printed cable flags. A wiring diagram shall be maintained as part of the build documentation. |
| AV-12 | `[NICE]` | A dedicated RGB LED indicator (visible from 30 m) shall reflect ArduPlane flight mode and arming status — useful during autonomous flight to confirm mode changes from the ground. |
| AV-13 | `[NICE]` | Downward-facing sensor port (belly, 60 × 60 mm) for optical flow or lidar altimeter module. |

> **💡 Advice — companion computer selection:** Start with a Raspberry Pi 4 (4 GB). The ecosystem for ROS2 + MAVLink (via MAVROS) is mature and well-documented. Jetson Nano adds GPU inference capability for vision tasks but runs hotter and draws more current. Orange Pi 5 is a cost-effective middle ground. All three fit in the 100 × 70 mm bay. Design the bay so the board can be removed without removing the flight controller.

---

## 9. Autopilot Parameter Management

> Each wing + tail combination is aerodynamically distinct. Flying a new configuration on the previous configuration's PID gains is dangerous and is the most common cause of maiden flight crashes on modular platforms.

| ID | Priority | Requirement |
|----|----------|-------------|
| AP-01 | `[MUST]` | Each wing + tail combination shall have its own ArduPlane `.param` file stored in version control. This file shall include at minimum: PID gains, ARSPD_FBW_MIN, ARSPD_FBW_MAX, STALL_PREVENTION, servo limits, and CG-derived TRIM_PITCH_CD. |
| AP-02 | `[MUST]` | ARSPD_FBW_MIN shall be set to ≥ 1.3 × calculated stall speed for the specific wing variant. This shall be recalculated for every new wing. |
| AP-03 | `[MUST]` | Each new wing + tail combination shall be validated in ArduPlane SITL using approximate aerodynamic parameters before physical maiden flight. |
| AP-04 | `[MUST]` | ArduPlane AUTOTUNE shall be run at minimum 80 m AGL for each new wing + tail combination, with a manual RC pilot maintaining altitude while AUTOTUNE operates. Auto-tuned parameters shall be saved to the configuration's `.param` file. |
| AP-05 | `[SHOULD]` | A ground station mission template shall be maintained for each configuration: loiter radius, cruise altitude, RTL altitude, and geofence boundary pre-set for the local flying site. |
| AP-06 | `[SHOULD]` | Parameter version history shall be maintained in a Git repository, with commit messages noting which configuration was flown and any anomalies observed. |
| AP-07 | `[NICE]` | A companion computer script shall verify that the loaded `.param` file matches the physically installed wing + tail combination (by reading a configuration ID from an NFC tag or QR code at the wing root) before arming is permitted. |

---

## 10. Electrical System

| ID | Priority | Requirement |
|----|----------|-------------|
| E-01 | `[MUST]` | Main power connector: XT60 or XT90. All batteries and harnesses shall use the same connector family. |
| E-02 | `[MUST]` | A physical arming switch shall be present, accessible from outside the fuselage, and shall cut motor power independently of the autopilot. |
| E-03 | `[MUST]` | All servo connectors at modular interfaces (wing root, tail boom) shall be keyed or labelled to prevent mis-wiring during reassembly. |
| E-04 | `[MUST]` | Wire gauge shall be sized for 1.5× maximum expected current: main power ≥ 14 AWG; servo rails ≥ 22 AWG. |
| E-05 | `[SHOULD]` | A current + voltage sensor shall be on the main battery lead, feeding the flight controller for real-time power monitoring and mAh consumed logging. |
| E-06 | `[SHOULD]` | All wiring runs shall be secured with cable ties or spiral wrap. No loose wiring in the airframe that could chafe against rotating parts or snag during assembly. |
| E-07 | `[SHOULD]` | The avionics bay shall have a dedicated power switch (independent of main battery) so avionics can be powered and configured with the motor electrically disconnected. |
| E-08 | `[NICE]` | A USB-C port accessible from outside the fuselage shall be connected to the companion computer for ground-based configuration and log download without opening the hatch. |

---

## 11. Safety Requirements

| ID | Priority | Requirement |
|----|----------|-------------|
| SF-01 | `[MUST]` | Manual RC override shall work at all times, in all autopilot modes, with no software path to disable it. |
| SF-02 | `[MUST]` | Loss of RC signal failsafe: RTL mode, triggered within 1 second of signal loss. |
| SF-03 | `[MUST]` | Loss of telemetry link shall not affect flight — autopilot continues current mission or RTL per pre-programmed failsafe. |
| SF-04 | `[MUST]` | Geofence shall be configured and active for all autonomous flights. Breach of geofence triggers RTL. |
| SF-05 | `[MUST]` | A pre-flight checklist shall be completed before every flight. A wing + tail swap requires an extended checklist including CG verification, V̄h/V̄v sign-off, and parameter file confirmation. |
| SF-06 | `[MUST]` | The propeller shall not be installed until the pilot is ready for taxi or launch and the field is clear. |
| SF-07 | `[SHOULD]` | A buzzer shall be connected to the flight controller and configured to sound on: arming, disarming, low battery, and GPS loss. |
| SF-08 | `[SHOULD]` | The CG shall be physically verified with a CG tool (or balance point test) before every maiden flight and after any weight change greater than 50 g. |
| SF-09 | `[NICE]` | An onboard buzzer or LED shall activate if the aircraft lands inverted or at an extreme attitude, to assist recovery location. |

---

## 12. Regulatory & Legal Requirements

> Regulations vary by country and change frequently. Verify current requirements with your national aviation authority before first flight.

| ID | Priority | Requirement |
|----|----------|-------------|
| R-01 | `[MUST]` | Register the aircraft with the relevant national authority if MTOW exceeds the registration threshold (250 g in EU/UK under EASA/CAA rules; 250 g in US under FAA rules). At 2.5–3.5 kg MTOW, registration is mandatory in virtually all jurisdictions. |
| R-02 | `[MUST]` | In the EU (EASA): the platform falls in **Open Category A2 or A3** depending on operational area. A2 requires an A2 CofC certificate. Flights near people require A2 compliance. |
| R-03 | `[MUST]` | Autonomous flights (beyond direct visual line of sight, or BVLOS) require specific authorisation in almost all jurisdictions. Do not fly autonomous missions BVLOS without explicit approval. |
| R-04 | `[MUST]` | Display a registration number visibly on the airframe. |
| R-05 | `[SHOULD]` | Carry third-party liability insurance for all flights. Many national aeroclub memberships include this; standalone policies are available. |
| R-06 | `[SHOULD]` | If exporting the platform or design files to certain countries, check dual-use export control regulations (EU Dual-Use Regulation, US EAR/ITAR). UAV technology is a controlled category in many regimes. |
| R-07 | `[NICE]` | If commercialising the platform for sale to research institutions, check CE marking requirements (EU) for electromagnetic compatibility and safety. |

> **💡 Advice:** In the Czech Republic (Prague), EASA rules apply. You are in Open Category. At MTOW > 900 g in residential areas you are likely in A3 (keep 150 m from residential areas) unless the design is A2-compliant. Check the CAA CZ (Úřad pro civilní letectví) for current rules. For research flights, consider applying for a Specific Category authorisation — it provides more flexibility for experimental operations.

---

## 13. Manufacturing & Repairability

| ID | Priority | Requirement |
|----|----------|-------------|
| MF-01 | `[MUST]` | Every 3D-printed structural part shall be an individual file, replaceable without reprinting adjacent parts. |
| MF-02 | `[MUST]` | All printed parts shall include version numbers in the model and on the print (embossed or debossed). |
| MF-03 | `[MUST]` | A bill of materials (BOM) shall list every component with: quantity, supplier, part number, unit cost, and lead time. |
| MF-04 | `[SHOULD]` | Fuselage sections shall be bolted, not permanently glued, wherever structural loads allow, to permit section replacement after crash damage. |
| MF-05 | `[SHOULD]` | All print files shall include recommended print settings: material, layer height, infill pattern, infill %, wall count, support strategy. |
| MF-06 | `[SHOULD]` | Carbon fibre components (tubes, rods) shall be sourced from a single supplier with consistent OD tolerance (±0.1 mm) to ensure spar pockets fit without sanding per build. |
| MF-07 | `[NICE]` | A crash damage matrix shall document: which parts are most likely to fail in a nose-in crash, a wingtip drag, and a hard belly landing — and the estimated print time to replace each. |

---

## 14. Testing & Validation Protocol

| ID | Priority | Requirement |
|----|----------|-------------|
| TV-01 | `[MUST]` | **Spar pull-out test:** each new wing variant shall be bench-tested at 2× design root bending moment before maiden flight. |
| TV-02 | `[MUST]` | **Boom bending test:** tail boom + socket shall withstand 2× max tail load at +6g without visible deformation. |
| TV-03 | `[MUST]` | **Control surface binding check:** full deflection of all control surfaces with airframe assembled shall show no binding, interference, or reduced travel. |
| TV-04 | `[MUST]` | **Motor run-up test (tethered):** full throttle static run for 60 seconds before maiden, checking for vibration, temperature, and thrust. |
| TV-05 | `[MUST]` | **SITL validation:** each new wing + tail combination shall be flown in ArduPlane SITL with approximate aerodynamic parameters and the configuration's `.param` file before any physical maiden. |
| TV-06 | `[MUST]` | **Maiden flight protocol:** first flight of any new configuration shall be manual RC, in FBWA (fly-by-wire A) mode, at safe altitude (minimum 50 m AGL), confirming control response and trim before engaging any autonomous mode. |
| TV-07 | `[SHOULD]` | **Weight and balance verification:** physical CG shall be measured with a CG tool after full assembly and compared to calculated value. Tolerance: ±5 mm. |
| TV-08 | `[SHOULD]` | **Vibration logging:** accelerometer data from the flight controller shall be reviewed after the first flight of each configuration to confirm vibration levels are within ArduPlane recommended limits (clipping < 1%, noise < 30 m/s²). |
| TV-09 | `[NICE]` | **Wind tunnel or CFD validation:** at least the primary wing variant should be validated against XFLR5 or OpenVSP predictions using flight data (airspeed, attitude, throttle) from a trim flight. |

---

## 15. Documentation Requirements

| ID | Priority | Requirement |
|----|----------|-------------|
| D-01 | `[MUST]` | Each wing + tail combination shall have a one-page configuration card: CG range, V̄h, V̄v, stall speed, cruise speed, ARSPD_FBW_MIN, and linked `.param` file. |
| D-02 | `[MUST]` | A pre-flight checklist shall be maintained as a printable document, with a dedicated section for wing + tail swap verification. |
| D-03 | `[MUST]` | A wiring diagram shall document every connector, cable run, and component in the fuselage. |
| D-04 | `[SHOULD]` | A build guide shall document assembly sequence with photos or renders at each major step. |
| D-05 | `[SHOULD]` | A flight log shall be maintained: date, configuration flown, battery cycles, anomalies, and any parameter changes. |
| D-06 | `[NICE]` | All documentation shall be maintained in a Git repository alongside CAD files and print files, with tagged releases for each stable configuration. |

---

## 16. Commercial & Open-Source Considerations

| ID | Priority | Requirement |
|----|----------|-------------|
| C-01 | `[OPEN]` | Decide on licence before publishing any files. Options: CC BY-SA (open, attribution required), CERN OHL (hardware-focused), or proprietary with paid licence. This decision affects all downstream commercial plans. |
| C-02 | `[SHOULD]` | The fuselage-to-wing interface standard (spar socket, bolt pattern, connector pinout) shall be published as a standalone specification document, versioned, to allow third-party wing development. |
| C-03 | `[SHOULD]` | If selling kits: prepare a per-unit cost model including print time, filament, CF stock, hardware, and packaging. Validate that target margins are achievable before committing to commercial production. |
| C-04 | `[NICE]` | Consider filing a design registration (not full patent — cheaper, faster) on the modular interface standard if commercialising. Patent is expensive; design registration provides meaningful deterrence for modest cost. |

---

## 17. Future / Optional Goals

| ID | Priority | Goal |
|----|----------|------|
| F-01 | `[NICE]` | Wing variant roadmap: v1 straight mid-wing (baseline) → v2 tapered/high-AR endurance → v3 swept sport → v4 delta. All using the same spar/bolt/connector interface. |
| F-02 | `[NICE]` | Twin-motor fuselage variant for differential thrust / engine-out research. Same wing and tail interfaces. |
| F-03 | `[NICE]` | Variable boom length set to allow tail arm tuning across widely different wing planforms. |
| F-04 | `[NICE]` | Configuration ID system: NFC tag or QR code at wing root + tail boom, read by companion computer to auto-load correct parameter file on connection. |
| F-05 | `[NICE]` | ROS2 integration: companion computer runs ROS2 nodes for mission planning, obstacle avoidance, and sensor fusion, bridged to ArduPlane via MAVROS. |
| F-06 | `[NICE]` | Swappable nose cone with standardised payload mount: camera, multispectral sensor, or lidar — same bolt pattern, different nose. |

---

## 18. Open Questions & Decisions Pending

These must be resolved before detailed design begins.

| ID | Question | Impact |
|----|----------|--------|
| OQ-01 | Spar tube size: 20 mm round or 15×15 mm square CF tube? | Freezes spar pocket geometry in fuselage — **highest priority decision** |
| OQ-02 | Pusher or tractor configuration? | Affects nose design, CG, prop clearance geometry |
| OQ-03 | Tail configuration: conventional T-tail, conventional low-tail, or V-tail for first variant? | Affects tail boom and fuselage rear design |
| OQ-04 | Landing gear: belly skid, nose+main wheels, or hand-launch + belly landing? | Affects fuselage underside geometry |
| OQ-05 | Wing skin method for first variant: printed ribs + monokote, fibreglass over foam, or fully printed panels? | Major impact on wing weight and build time |
| OQ-06 | Commercial licence vs open-source? | Must be decided before publishing any files publicly |
| OQ-07 | Target flying site and local regulations (Czech EASA Open/Specific category)? | Affects geofence parameters, MTOW, and operational limits |

---

## Appendix A — Key Formulae Reference

```
Horizontal tail volume coefficient:   V̄h = (Sh × Lh) / (S × c̄)     target: 0.35 – 0.45
Vertical tail volume coefficient:     V̄v = (Sv × Lv) / (S × b)      target: 0.04 – 0.06
Static longitudinal margin:           SM = (NP − CG) / MAC            target: 5% – 15%
Wing loading:                         WL = MTOW / S                   target: ≤ 35 g/dm²
Stall speed:                          Vs = sqrt(2 × MTOW / (ρ × S × CLmax))
Min autopilot airspeed:               ARSPD_FBW_MIN ≥ 1.3 × Vs
```

Where:
- `Sh` = horizontal stabiliser area (dm²)
- `Sv` = vertical fin area (dm²)
- `Lh` = distance from wing AC to horizontal tail AC (m)
- `Lv` = distance from wing AC to vertical tail AC (m)
- `S`  = wing area (dm²)
- `c̄` = mean aerodynamic chord (m)
- `b`  = wingspan (m)
- `NP` = neutral point position (% MAC from LE)
- `CG` = centre of gravity position (% MAC from LE)

---

## Appendix B — Recommended Tools & Software

| Category | Tool | Notes |
|----------|------|-------|
| Aerodynamic analysis | XFLR5 | Free. Airfoil + full aircraft analysis. Use for NP, Cm, Cl/Cd. |
| 3D geometry / CFD | OpenVSP | Free, NASA-developed. Good for volume coefficient calculations. |
| Autopilot simulation | ArduPlane SITL | Simulate new configurations before maiden flight. |
| Ground station | Mission Planner / QGroundControl | Both free. Mission Planner has deeper ArduPlane integration. |
| CAD | Fusion 360 / FreeCAD / OnShape | Fusion 360 free for hobbyists/startups under revenue threshold. |
| Parameter management | Git + `.param` files | Version control for every configuration. |
| Weight budget | Spreadsheet (Excel / Calc) | Simple but essential. Track actuals vs estimates from day one. |

---

*Document version 0.1 — generated during initial design requirements phase.*  
*All requirements subject to revision as design matures.*
