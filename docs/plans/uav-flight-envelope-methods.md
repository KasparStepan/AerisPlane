# Flight Performance Envelope Methods for Small Electric RC/UAV Fixed‑Wing Aircraft

## Executive Overview

This report summarizes a consistent set of closed‑form equations and conceptual‑design methods for the flight‑performance envelope of small electric fixed‑wing RC/UAV aircraft (≈1–20 kg), assuming an available aerodynamic solver (VLM), weight build‑up, ISA atmosphere, and basic electric propulsion models. The treatment follows classical performance texts (Anderson, McCormick, Raymer, Drela/MIT) and RC‑specific notes, with adaptations appropriate to low Reynolds numbers and off‑the‑shelf BLDC/propeller components.[web:2][web:3][web:4][web:55][web:59]

The focus is point performance (steady level, climb, glide) and derived envelope quantities (characteristic speeds, climb ceilings, glide ratios, electric endurance/range) at conceptual‑design fidelity (≈10–15 % accuracy) without CFD or full BEMT.

---

## 1 Power Required and Power Available

### 1.1 Drag and power required in level flight

Assume a parabolic drag polar in terms of lift coefficient:
\[
C_D = C_{D0} + k C_L^2 \quad \text{with} \quad k = \frac{1}{\pi ARe}
\]  
for an aspect ratio \(AR\) and Oswald efficiency \(e\).[web:2][web:3]

For steady, level, unaccelerated flight:
\[
L = W, \quad L = \tfrac12 \rho V^2 S C_L, \quad D = \tfrac12 \rho V^2 S C_D
\]
so
\[
C_L = \frac{2W}{\rho V^2 S}
\]
and the drag as a function of speed can be written
\[
D(V) = \tfrac12 \rho V^2 S C_{D0} + \frac{2 k W^2}{\rho V^2 S}
\]
showing the usual parasitic \(\propto V^2\) and induced \(\propto 1/V^2\) components.[web:2][web:3]

Power required for level flight is
\[
P_R(V) = D(V) V = \tfrac12 \rho S C_{D0} V^3 + \frac{2 k W^2}{\rho S} \frac{1}{V}
\]
This is the standard \(P_R\) curve with a minimum at an intermediate speed; it directly follows Anderson and MIT performance notes.[web:2][web:3][web:9]

For implementation, you can either:
- Use the drag‑polar form above (with \(C_{D0}, k\) estimated from geometry or fitted to VLM results), or
- Evaluate CL and CD via your VLM+profile model at each \(V\), then compute \(D = \tfrac12 \rho V^2 S C_D\) and \(P_R = DV\).

### 1.2 Power available from electric propulsion

#### 1.2.1 Standard propeller coefficient model

Small propellers are conventionally characterized via thrust and power coefficients as functions of advance ratio \(J\):
\[
J = \frac{V}{n D}, \quad C_T(J) = \frac{T}{\rho n^2 D^4}, \quad C_P(J) = \frac{P_{shaft}}{\rho n^3 D^5}
\]
where \(n\) is rev/s and \(D\) is diameter.[web:20][web:22][web:28]

Propeller efficiency is then
\[
\eta_p(J) = \frac{T V}{P_{shaft}} = \frac{C_T(J) J}{C_P(J)}
\][web:20][web:22]

Given a battery bus voltage \(V_b\), motor+ESC efficiency \(\eta_m\), and propeller map \(C_P(J)\), a simple conceptual model for power available at a given airspeed is
\[
P_A(V) = \eta_m(V)\, P_{elec,max}\, f_{throttle} \approx \eta_m(V)\, \eta_p(J)\, P_{shaft,max}
\]
with \(P_{shaft,max}\) limited by motor current and thermal limits.[web:2][web:3][web:38]

In practice for conceptual design:
- Choose a design RPM (or \(n\)) at which the motor can continuously deliver \(P_{shaft}\) within current limits.
- From \(V\), compute \(J\); from \(J\), interpolate \(C_T, C_P\) from a database (e.g. UIUC Propeller Database).[web:28][web:29]
- Compute \(P_{shaft}(V) = C_P(J) \rho n^3 D^5\) and \(T(V) = C_T(J) \rho n^2 D^4\).
- Optionally include a mild reduction of \(P_{shaft,max}\) with altitude via motor cooling limits.

#### 1.2.2 Simplified CT/CP models for conceptual design

The UIUC propeller tests for typical APC/Aero‑Naut/Graupner props show \(C_T(J)\) and \(C_P(J)\) that are smooth and well approximated by low‑order polynomials on the useful \(J\) range.[web:29][web:40][web:42]
A simple generic fit that works reasonably across many RC props is:
\[
C_T(J) \approx a_0 + a_1 J + a_2 J^2, \quad
C_P(J) \approx b_0 + b_1 J + b_2 J^2
\]
with coefficients calibrated from 2–4 points (e.g. static test \(J=0\) plus one or two wind‑tunnel / manufacturer data points).[web:28][web:38]

For "unknown" props, UIUC data indicate peak propeller efficiencies \(\eta_p\) in the range 0.5–0.7 for small electric props; you can enforce this by choosing coefficients such that \(\eta_p(J)\) peaks near \(J \approx 0.4\text{–}0.7\) at the desired \(\eta_{p,max}\).[web:38][web:42]

#### 1.2.3 Motor‑limited power available

For a BLDC motor with Kv [rpm/V], resistance \(R_m\), and no‑load current \(I_0\), the no‑load speed is approximately \(\omega_0 = K_v V_b\) (in rad/s with appropriate unit conversion).[web:73][web:74]
Under load, a simple DC model gives
\[
I = I_0 + \frac{V_b - K_e \omega}{R_m}, \quad T_m = K_t (I - I_0)
\]
with \(K_e = 1/K_v\) and \(K_t = 60/(2\pi K_v)\) in SI units for Kv in rpm/V.[web:74][web:77]

Shaft power is \(P_{shaft} = T_m \omega\), electrical power \(P_{elec} = V_b I\), and motor efficiency \(\eta_m = P_{shaft}/P_{elec}\).[web:70][web:73]

For conceptual design, one can assume:
- A nearly flat \(P_{shaft,max}\) up to some RPM limit, then cap by voltage back‑EMF and ESC current limit.
- \(\eta_m\) ≈ 0.8–0.9 near the design operating point for a well‑sized motor.[web:70][web:83]

Then
\[
P_A(V) \approx \eta_p(J) \eta_m P_{elec,max}(h)
\]
where \(P_{elec,max}\) is bounded by battery current and motor limits.

### 1.3 Altitude effects on power curves

In the parabolic‑polar approximation with constant \(C_{D0}, k\), the power‑required minimum increases roughly linearly with \(1/\sqrt{\rho}\) while the corresponding speed increases with \(\rho^{-1/2}\).[web:2][web:3]
From the analytical form of \(P_R(V)\), the minimum scales as
\[
P_{R,min} \propto \frac{W^{3/2}}{\sqrt{\rho S}} \sqrt{C_{D0} k}
\]
so at higher altitude (lower \(\rho\)) the entire \(P_R\) curve shifts upward and to the right in \(V\).[web:2][web:3]

For propellers, thrust and shaft power scale as \(\rho\) for fixed RPM and \(J\):
\[
T \propto \rho, \quad P_{shaft} \propto \rho
\]
so the \(P_A\) curve also scales with \(\rho\) (ignoring modest electrical‑cooling limits), tending to reduce excess power and climb capability with altitude.[web:20][web:42]

---

## 2 Characteristic Speeds

### 2.1 Stall speed and CL_max estimation

The classical 1‑g stall speed for level flight is
\[
V_{stall} = \sqrt{\frac{2W}{\rho S C_{L,max}}}
\]

When your aerodynamic solver does not directly provide \(C_{L,max}\), options include:

1. **Empirical airfoil data at low Reynolds number** (Selig "Airfoils at Low Speeds" and UIUC low‑speed airfoil data) for representative sections at your Reynolds number.[web:53][web:47][web:44]
2. **RC model design notes** providing typical 3‑D wing \(C_{L,max}\) values and a 3‑D/2‑D reduction factor (≈0.9 for AR>5 unswept wings).[web:57][web:65]
3. **Back‑solving from validated RC designs** with known stall speeds.

RC‑oriented sources give the following indicative ranges at RC Reynolds numbers \(\approx 1\times 10^5 \text{–} 5\times 10^5\):[web:62][web:64][web:65]
- Mildly cambered / flat‑bottom high‑lift airfoils (e.g. S1223‑type): \(C_{L,max,2D} \approx 1.8\text{–}2.2\); 3‑D wing \(C_{L,max} \approx 1.5\text{–}1.9\) depending on AR and flap use.
- Typical highly cambered RC lifting sections (without flaps): \(C_{L,max} \approx 1.4\text{–}1.6\).[web:62][web:64]
- Semi‑symmetric sport airfoils: \(C_{L,max} \approx 1.2\text{–}1.4\).[web:62]
- Symmetric sections (e.g. NACA 00xx) at low Re: \(C_{L,max} \approx 1.1\text{–}1.3\) due to laminar separation effects.[web:44][web:50]

A pragmatic conceptual‑design approach is:
1. Pick a representative airfoil from low‑Re databases (UIUC, Selig LSAT) with similar thickness/camber.[web:47][web:53]
2. Read off \(C_{L,max,2D}\) around the design Reynolds number (or interpolate).[web:47][web:53]
3. Apply a 3‑D correction \(C_{L,max,3D} \approx 0.9 C_{L,max,2D}\) for unswept AR ≳ 6 RC wings.[web:57][web:65]

### 2.2 Minimum drag speed (max L/D, best range)

With \(C_D = C_{D0} + k C_L^2\), lift‑to‑drag ratio is
\[
\frac{L}{D} = \frac{C_L}{C_{D0} + k C_L^2}
\]
Maximizing \(L/D\) gives the well‑known condition
\[
C_{L,*} = \sqrt{\frac{C_{D0}}{k}}, \quad C_{D,*} = 2 C_{D0}, \quad \left(\frac{L}{D}\right)_{max} = \frac{1}{2\sqrt{C_{D0} k}}
\][web:2][web:3]

Using \(C_L = 2W/(\rho V^2 S)\), the corresponding speed is
\[
V_{L/D\,max} = V_* = \sqrt{\frac{2W}{\rho S C_{L,*}}} = \left( \frac{2W}{\rho S} \right)^{1/2} \left( \frac{k}{C_{D0}} \right)^{1/4}
\][web:2][web:3]
This is simultaneously the minimum‑drag speed and, for propeller aircraft, the speed associated with best range in fuel‑burning cases (see §5).

### 2.3 Minimum power speed (best endurance)

For a propeller aircraft characterized by power rather than thrust, minimum power required corresponds to maximum \(C_L^{3/2}/C_D\).[web:3][web:9]
With the parabolic drag polar one finds
\[
\left( \frac{C_L^{3/2}}{C_D} \right)_{max} \quad \Rightarrow \quad C_{L,mp} = \sqrt{3} C_{L,*}
\]
leading to
\[
V_{mp} = 3^{-1/4} V_* \approx 0.76 V_*
\][web:3][web:9]

This is the speed for minimum \(P_R\), hence best endurance for a propeller or electric aircraft with nearly constant shaft power capability.[web:3][web:9]

### 2.4 Maximum speed

Maximum level speed \(V_{max}\) at a given altitude satisfies
\[
P_A(V_{max}) = P_R(V_{max})
\]
Given the strong dependence of \(P_R\) on \(V\) and modest variation of \(P_A\) with \(V\) (prop power roughly flat over its efficient range), this is typically solved numerically by scanning over speed.[web:2][web:3][web:5]

In a conceptual tool, one can:
1. Compute \(P_R(V)\) across a speed grid using the drag polar (or direct VLM results).
2. Compute \(P_A(V)\) using an assumed \(\eta_p(J)\) and motor power cap.
3. Take \(V_{max}\) as the largest speed where \(P_A \ge P_R\).

### 2.5 Typical speed ranges for 1–5 kg RC aircraft

RC design notes and heavy‑lift competition reports indicate typical stall and cruise speeds:[web:62][web:64][web:59]
- Stall speeds \(V_{stall}\) on the order of 20–35 mph (≈9–16 m/s) for high‑lift, lightly loaded aircraft with \(C_{L,max} \approx 1.5\text{–}2.0\).
- Cruise and best‑range speeds usually 1.3–1.7 times stall, i.e. ≈25–60 mph (≈11–27 m/s) for 1–5 kg models.
- Pylon and racing models with thinner sections and higher wing loading can cruise well above 40–50 m/s, but these are outliers for small UAVs.[web:61][web:64]

These ranges are consistent with Reynolds numbers \(\approx 1\times 10^5 \text{–} 5\times 10^5\) for chords in the 0.15–0.3 m range and speeds in the tens of m/s.[web:44][web:53]

---

## 3 Rate of Climb and Ceilings

### 3.1 Excess‑power method and climb drag

Specific excess power is
\[
SEP = \frac{P_A - P_R}{W}
\]
and the climb rate at flight path angle \(\gamma\) is
\[
\text{ROC} = \dot{h} = V \sin\gamma = SEP
\]
for steady climb (no change in kinetic energy).[web:9][web:5]

A common approximation is to use level‑flight drag at the climb speed in \(P_R\) and ignore the change in drag due to finite \(\gamma\). A more exact treatment notes that the required thrust in climb is
\[
T = D + W \sin\gamma
\]
so the power required is
\[
P_{R,climb} = T V = D V + W V \sin\gamma = P_R + W \dot{h}
\]
Substituting \(\dot{h} = (P_A - P_{R,climb})/W\) gives back the same SEP expression; to first order, using level‑flight drag at the same \(V\) is sufficient for small climb angles.[web:5][web:9]

For practical UAV climb performance, the dominant driver is **excess shaft power**; induced‑drag changes with \(\gamma\) are a second‑order effect for \(|\gamma| \lesssim 10^\circ\).

### 3.2 Maximum rate of climb and its speed

For a propeller/electric aircraft where \(P_A\) is weakly dependent on \(V\), the rate of climb is
\[
\dot{h}(V) = \frac{P_A(V) - P_R(V)}{W}
\]
The maximum ROC at a given altitude typically occurs near the minimum of \(P_R\), i.e. around \(V_{mp}\), since \(P_A\) is approximately flat there.[web:3][web:9]

Algorithmically:
1. For each altitude, compute \(P_R(V)\) and \(P_A(V)\) over your speed grid.
2. Compute \(\dot{h}(V)\) and select the maximum; store \(\dot{h}_{max}(h)\) and the corresponding \(V_{y}(h)\).
3. Plot \(\dot{h}_{max}\) vs altitude to visualize climb capability and approach to ceiling.

### 3.3 Service and absolute ceiling for electric UAVs

Textbook performance defines:[web:2][web:9]
- **Service ceiling**: altitude at which maximum ROC drops to 100 ft/min (≈0.5 m/s) for manned aircraft.
- **Absolute ceiling**: altitude where \(\dot{h}_{max} = 0\); level flight is possible but no climb.

For small UAVs, speeds and mission profiles differ, but the same definitions can be adopted with modified ROC thresholds. Many small‑UAV studies still use 0.5 m/s as a practical service‑ceiling ROC threshold, though for very small or VTOL vehicles values as low as 0.3 m/s are sometimes employed.[web:38][web:42]

Given \(\dot{h}_{max}(h)\) from the excess‑power calculation:
- **Absolute ceiling** is the highest altitude where \(P_A(V) \ge P_R(V)\) for some \(V\), i.e. where \(\max_V \dot{h}(V) = 0\) (numerically: last positive value).
- **Service ceiling** for a chosen ROC threshold \(ROC_{serv}\) is the altitude where \(\dot{h}_{max}(h) = ROC_{serv}\).

Because both \(P_R\) and \(P_A\) scale with \(\rho\), ceilings reflect a detailed balance between propeller performance (\(\rho\) scaling, off‑design \(J\)), airframe drag (induced \(\propto 1/\rho\)), and motor cooling/current limits with altitude.[web:2][web:28][web:33]

---

## 4 Glide Performance

### 4.1 Best glide speed and glide ratio

In power‑off glide, thrust is zero and equilibrium along the flight path gives
\[
W \sin\gamma = D, \quad W \cos\gamma = L
\]
so
\[
\tan\gamma = \frac{D}{L} = \frac{1}{L/D}
\]
Maximum (most negative) glide ratio \(L/D\) corresponds to minimum sink angle magnitude; it occurs at the same \(C_L\) and \(V\) as \(L/D_{max}\) in powered level flight, as long as the drag polar is unchanged.[web:3][web:5]

Thus best‑glide speed and \((L/D)_{max}\) are
\[
V_{BG} = V_* = V_{L/D\,max}, \quad \left(\frac{L}{D}\right)_{BG} = \left(\frac{L}{D}\right)_{max}
\]
with \(V_*\) from §2.2.

### 4.2 Glide range from altitude

Neglecting winds and assuming constant \(L/D\), glide path geometry gives
\[
\frac{\text{horizontal distance}}{\text{altitude lost}} = \frac{L}{D}
\]
so the still‑air range from altitude \(h_0\) at best glide is approximately
\[
R_{BG} \approx \left(\frac{L}{D}\right)_{max} h_0
\][web:3][web:5]

If the vehicle glides at a non‑optimal \(C_L\) (e.g. to meet speed constraints), use the corresponding \(L/D\) from your drag polar instead. Straight glide vs "best‑range" glide differ only by the chosen \(L/D\).

### 4.3 Minimum sink rate speed

Minimum sink in glide corresponds to **minimum vertical speed**, not maximum horizontal distance. For a rigid‑wing glider with parabolic drag polar, the minimum sink speed occurs at the \(C_L\) that maximizes \(C_L^{3/2}/C_D\), exactly as for minimum power in powered flight.[web:3][web:9]

Consequently:
\[
V_{min\,sink} = V_{mp} = 3^{-1/4} V_* \approx 0.76 V_*,
\]
with glide at \(C_{L,mp} = \sqrt{3} C_{L,*}\), yielding a sink rate (negative \(\dot{h}\)) proportional to \(P_{R,min}/W\).[web:3][web:9]

---

## 5 Electric Endurance and Range

### 5.1 Maximum endurance (constant weight electric)

For a battery‑electric aircraft, weight is nearly constant over the mission. Let the usable battery energy be \(E_b\) (J), and let total propulsive efficiency (motor+prop) at operating condition be \(\eta_p\). Then time aloft in steady flight at power \(P_R(V)\) is
\[
E = \frac{E_b \, \eta_p(V)}{P_R(V)}
\]
Maximum endurance is achieved by minimizing \(P_R/\eta_p\). If \(\eta_p\) is nearly flat vs speed over the efficient prop range (a reasonable approximation for RC props over a modest speed band), this reduces to flying at minimum power required \(V_{mp}\) from §2.3:[web:8][web:9]
\[
E_{max} \approx \frac{E_b \, \eta_{p,mp}}{P_{R,min}}
\]
with \(P_{R,min}\) computed from the drag polar and \(V_{mp}\).[web:3][web:8]

### 5.2 Maximum range for electric

Range is horizontal distance covered:
\[
R = \int_0^E V(t) \; dt
\]
For constant weight and operation at a fixed speed, this simplifies to
\[
R = V \frac{E_b \, \eta_p(V)}{P_R(V)} = \frac{E_b \, \eta_p(V)}{W} \frac{L}{D}
\]
using \(P_R = DV = (W L/D)(V/L)\) and \(L \approx W\) in steady level flight.[web:8][web:3]

Thus, for constant weight electric aircraft, maximum range occurs at the condition maximizing \(\eta_p (L/D)\). If \(\eta_p\) is weakly dependent on speed near \(V_*\), this is approximately the same as maximizing \(L/D\), i.e. flying at \(V_*\):[web:8][web:9]
\[
R_{max} \approx \frac{E_b \, \eta_{p,*}}{W} \left( \frac{L}{D} \right)_{max}
\]
The simplified formula
\[
R \approx E_b \, \eta \frac{(L/D)_{max}}{W}
\]
is therefore appropriate for conceptual sizing of electric UAV range, provided a realistic \(\eta_p\) is used and operation is near \(L/D_{max}\).[web:8][web:9]

### 5.3 Modeling propulsive efficiency versus speed

Propeller efficiency curves from UIUC and other low‑Re tests show a strong dependence on advance ratio \(J\): low at very small \(J\) (static or near‑static), rising to a peak at moderate \(J\), then declining at high \(J\) as the prop unloads.[web:20][web:42][web:38]

A conceptual model suitable for your tool is:
- Represent \(\eta_p(J)\) as a quadratic or cubic function peaking at a prescribed \(J_{opt}\) with peak efficiency \(\eta_{p,max}\) based on representative data (e.g. \(\eta_{p,max} \approx 0.55\text{–}0.7\) at \(J_{opt} \approx 0.4\text{–}0.7\) for APC/Graupner electric props).[web:20][web:38]
- Map airspeed to \(J\) using the current RPM estimate.
- In endurance and range calculations, evaluate \(\eta_p\) at the chosen operating point (\(V_{mp}\) or \(V_*\)).

This adds only one state (RPM or \(J\)) and keeps the model within conceptual‑design complexity while capturing the main effects of propulsive efficiency on performance.

---

## 6 Propeller Efficiency Modeling

### 6.1 Standard thrust and power coefficient model

The widely used non‑dimensional propeller performance definitions are:[web:20][web:22][web:38]
\[
C_T = \frac{T}{\rho n^2 D^4}, \quad
C_P = \frac{P_{shaft}}{\rho n^3 D^5}, \quad
J = \frac{V}{n D}, \quad
\eta_p = \frac{C_T J}{C_P}
\]
Wind‑tunnel campaigns at UIUC and elsewhere have generated \(C_T(J)\), \(C_P(J)\), and \(\eta_p(J)\) curves for hundreds of small RC propellers across Reynolds numbers and RPMs.[web:28][web:29][web:39][web:40]

For conceptual calculations:
- Use generic \(C_T(J)\), \(C_P(J)\) polars from a prop of similar diameter and pitch (e.g. from the UIUC database) as proxies.[web:29][web:33]
- Scale thrust and power according to \(D^4\) and \(D^5\) where appropriate when switching prop sizes, recognizing that detailed blade geometry still matters.[web:20][web:42]

### 6.2 Empirical CT(J) and CP(J) fits

Analysis of the UIUC data and follow‑on studies shows that, over the operating \(J\) range for cruise (roughly 0.3–0.8), both \(C_T\) and \(C_P\) are nearly linear or gently curved functions of \(J\).[web:40][web:42]
A simple empirical representation that balances fidelity and simplicity is:
\[
C_T(J) = c_{T0} + c_{T1} J + c_{T2} J^2, \quad
C_P(J) = c_{P0} + c_{P1} J + c_{P2} J^2
\]
with coefficients determined by least‑squares fits to the UIUC data for a chosen reference propeller or family.[web:28][web:37]

Where no test data are available, one can use typical values drawn from UAV‑prop studies:
- At the efficiency peak \(J_{opt}\), \(C_T \approx 0.08\text{–}0.12\), \(C_P \approx 0.05\text{–}0.08\), giving \(\eta_p \approx 0.5\text{–}0.7\).[web:20][web:42]

### 6.3 Momentum/BEMT for conceptual design

Momentum theory and blade‑element‑momentum theory (BEMT) are the foundation for many propeller analysis codes (e.g. QPROP, PROPID) and are routinely applied to small UAV and RC props.[web:22][web:26][web:21]

However:
- Full BEMT requires detailed blade geometry (chord, twist, airfoil, Re distribution) and iteration for induced velocities, which is overkill for early conceptual sizing.
- For fixed‑pitch, off‑the‑shelf propellers, measured \(C_T, C_P\) curves from databases such as UIUC already collapse the detailed aerodynamics into usable coefficients.[web:28][web:40]

A practical compromise is:
- Use published \(C_T(J), C_P(J)\) for representative props (or your own static bench data) in conceptual tools.
- Reserve BEMT for later design stages or when designing custom blades, or when extrapolating far beyond available data (e.g. unusual Reynolds numbers or inflow angles).

### 6.4 Published small‑propeller datasets

Key datasets and tools for small RC/UAV propellers include:
- **UIUC Propeller Database** (Volumes 1–4): ≈250 off‑the‑shelf props tested with thrust/torque vs \(J\) and \(C_T, C_P, \eta_p\) curves.[web:28][web:29][web:30][web:39]
- **Brandt & Selig “Propeller Performance Data at Low Reynolds Numbers”**: foundational AIAA paper and MSc thesis documenting low‑Re prop tests.[web:29][web:40]
- **PropDBTools** (open‑source): utilities to parse and use UIUC prop data programmatically.[web:37][web:35]
- Numerous follow‑up experimental rigs and rolling‑rig tests that confirm typical shapes of \(C_T(J), C_P(J)\) and efficiency curves for UAV props.[web:38][web:42]

These resources are ideal reference datasets for fitting generic polars or calibrating simplified models in a Python‑based conceptual design tool.

---

## 7 Motor–Propeller Matching

### 7.1 Operating point (RPM, current, torque) at given speed and throttle

For a given BLDC motor, propeller, airspeed, and throttle command, the steady operating point is determined by the intersection of:
1. The **motor characteristic** (DC equivalent): voltage, speed, and torque/current relation.
2. The **propeller load curve**: torque vs RPM at the given airspeed.

A common DC motor model is:[web:73][web:74][web:77]
\[
V_{esc} = u_{th} V_b, \quad
I = I_0 + \frac{V_{esc} - K_e \omega}{R_m}, \quad
T_m = K_t (I - I_0)
\]
where \(u_{th} \in [0,1]\) is the throttle duty cycle, \(K_e = 1/K_v\), and \(K_t = 60/(2\pi K_v)\) in SI if Kv is in rpm/V.

For the propeller, torque coefficient \(C_Q\) (often denoted \(C_P/(2\pi)\) up to constants) gives
\[
Q_p = C_Q(J) \rho n^2 D^5
\]
relating required torque to RPM at a given advance ratio \(J\).[web:20][web:22]

The steady operating point satisfies
\[
T_m(\omega, u_{th}) = Q_p(\omega, V)
\]
This non‑linear equation can be solved numerically in your code (e.g. 1‑D root find over \(\omega\)) to obtain \(\omega\) and then \(I, T, P_{shaft}\) for each \(V, u_{th}\) pair.

Where detailed \(C_Q(J)\) is unavailable, the propeller torque can be approximated by a quadratic in RPM, calibrated from static thrust/torque measurements or manufacturer curves.[web:76][web:82]

### 7.2 Electrical and mechanical power, efficiency

Once the operating point is known:
\[
P_{elec} = V_b I, \quad
P_{shaft} = T_m \omega, \quad
\eta_m = \frac{P_{shaft}}{P_{elec}}
\]
Propulsive efficiency \(\eta_p\) is computed from \(C_T, C_P, J\) or directly from thrust and shaft power as
\[
\eta_p = \frac{T V}{P_{shaft}}
\][web:20][web:22][web:83]

Total shaft‑to‑thrust efficiency is then \(\eta_{tot} = \eta_m \eta_p\), which feeds back into the performance, endurance, and range calculations above.

### 7.3 Throttle–voltage mapping in ESCs

Most RC ESCs for BLDC motors modulate **duty cycle** of a fixed DC bus voltage using PWM, presenting the motor windings with an effective phase voltage proportional to throttle command at low to medium throttle.[web:73][web:78]

For conceptual modeling it is reasonable to approximate:
\[
V_{esc} \approx u_{th} V_b
\]
up to saturation regions near maximum duty where commutation and timing advance effects appear.[web:73][web:78]

Your matching algorithm can then:
1. For each throttle \(u_{th}\), compute \(V_{esc}\) and the motor line (speed–torque relation).
2. Intersect with the propeller torque curve at the desired airspeed.
3. Check that current and power remain within limits; if not, back off \(u_{th}\).

---

## 8 Altitude and Reynolds‑Number Effects

### 8.1 Reynolds number reduction and CD0 for small UAVs

For small UAVs, chord Reynolds numbers are often in the range \(10^5\)–\(5\times 10^5\). In this regime, airfoil performance is highly sensitive to Reynolds number; both \(C_{L,max}\) and \(L/D\) degrade as Re decreases, primarily through increased profile drag and more prominent laminar separation bubbles.[web:44][web:53][web:52]

Key trends from low‑Re wind‑tunnel data include:[web:44][web:53][web:52]
- Maximum lift coefficient decreases by \(\mathcal{O}(10\text{–}20\%)\) when Re drops from ≈300,000 to ≈100,000 for many thin sections.
- Minimum \(C_D\) increases and the low‑drag "bucket" narrows at very low Re.
- Symmetric sections are particularly prone to non‑linear lift curves and hysteresis at very low Re.

For conceptual design across altitude, it is therefore reasonable to:
- Treat \(C_{D0}\) as weakly increasing with decreasing Re below ≈200,000, based on airfoil data tables.[web:47][web:53]
- Optionally incorporate a Re‑dependent correction factor to \(C_{D0}\) from a representative low‑Re airfoil polar in your database.

Given your VLM framework, a practical approach is:
1. At each altitude and speed, compute Re at reference chord.
2. Query 2‑D airfoil data for that Re to obtain sectional \(c_d\) and integrate/span‑average through your VLM method.
3. Use the resulting effective \(C_D\) in the drag polar instead of assuming constant \(C_{D0}\).

### 8.2 Propeller performance at altitude

For a given RPM and prop geometry, the non‑dimensional \(C_T(J)\) and \(C_P(J)\) are only weakly dependent on density; the dominant effect is the \(\rho\) scaling in thrust and power definitions:[web:20][web:42][web:33]
\[
T \propto \rho, \quad P_{shaft} \propto \rho
\]
Thus, at higher altitude (lower \(\rho\)) and same RPM and \(J\):
- Thrust and shaft power decrease roughly in proportion to \(\rho\).
- Propeller efficiency \(\eta_p(J) = C_T J / C_P\) is nearly unchanged, as \(C_T, C_P\) are dimensionless.[web:20][web:42]

Secondary effects include:
- Change in Reynolds number at the blade sections (proportional to \(\rho V_{local}\)), which slightly alters sectional lift/drag and hence \(C_T, C_P\).[web:33][web:42]
- Possible motor‑cooling limitations due to reduced convective heat transfer at low \(\rho\), leading to reduced allowable continuous current.

In a conceptual tool, altitude effects on the propeller can therefore be modeled by:
1. Scaling thrust and power with density: \(T(h) = T_{SL} \rho(h)/\rho_{SL}\), \(P_{shaft}(h) = P_{SL} \rho(h)/\rho_{SL}\) at given RPM and \(J\).[web:20][web:42]
2. Optionally applying a small Re‑dependent correction to \(C_T, C_P\) based on propeller tests at different RPMs (e.g. UIUC data).[web:33][web:42]

---

## 9 Implementation Notes for a Python Conceptual Tool

For the 1–20 kg RC/UAV class and the capabilities you already have (ISA atmosphere, VLM, weight build‑up, motor/prop/battery classes), the following implementation strategy keeps complexity moderate while achieving ≈10–15 % accuracy:

- Use a **parabolic drag polar** at the aircraft level, but obtain \(C_{D0}\) and \(k\) from span‑averaged VLM+profile computations at representative speeds and loading rather than pure handbook estimates.[web:2][web:3][web:57]
- Model **power required** via \(P_R(V) = DV\), using drag from your solver; optionally fit an analytic \(P_R(V)\) curve for speed.
- Obtain **propeller performance** from a small database of representative \(C_T(J), C_P(J)\) polars (e.g. 5–10 common APC electric props) extracted from the UIUC database, and use scaled versions for similar geometries.[web:28][web:29][web:37]
- Implement a simple **motor–prop matching** loop solving \(T_m(\omega) = Q_p(\omega, V)\) for \(\omega\) at each (altitude, speed, throttle), using the BLDC DC‑equivalent model.
- Compute **characteristic speeds**, climb, glide, endurance, and range directly from these curves using the analytic relationships summarized in §§2–5.
- Incorporate **altitude and Reynolds effects** by recomputing aerodynamic and prop coefficients as functions of \(\rho(h)\) and Re, at least through scaling and look‑ups in airfoil/propeller tables.

This architecture is well aligned with methods presented in Anderson, McCormick, and Drela’s performance notes, and with RC‑specific guidance from Selig and Nicolai, while remaining lightweight enough for iterative conceptual design and optimization.[web:2][web:3][web:4][web:28][web:57]
