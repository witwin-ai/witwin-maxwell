# Direct tails retaining the exact first-wave coefficient

Implemented explicit free-space direct_first_wave_min_v1 following the actual
direct_positive_tail_min_v1 growing-ka failure and its saved worst geometry.
A single witness does not establish full admission.
Original geometry, p6..12, thresholds and resource gates remain unchanged.

The reviewed exact-first-wave majorant is
Phi*(u)=(1+Ku)R1(u)+C(u)(exp(Ku)-1-Ku),
R_m=(1-u)^(-m), C=(R1+R2)/2, K=|k|D.
The reviewed direct positive majorant was Psi=C exp(Ku), giving direct
source/target tails from T1/T2/T3. Their coefficientwise difference is

Delta=Psi-Phi*=u R2(u)(1+Ku)/2,
Delta'=R2/2+K u R2+u R3+K u^2 R3.

All coefficients of Delta and Delta' are nonnegative, so exact direct tails
for Phi* and Phi*' equal the previous direct tails minus these positive
correction tails. Phi* and Phi*' also retain nonnegative coefficients by
the previously reviewed finite-wave first-coefficient proof; the resulting
difference is therefore an upper majorant for the real derivative norm.
No coefficient roundoff, Ewald or physical certificate is added.

For held a, omitted b, P=p+1, define U_m(T) as the degree>=T tail in b
of (1-a-b)^(-m). For T>=0 it is
 U_m(T)=q^T L^(-m) A_m(T,q), L=1-a, q=b/L,
with A_m from the reviewed shifted-geometric sums, including T=0 (whole
series). Products by b^j select U_m(max(P-j,0)). Thus correction tails are

DG=[a U2(P)+b U2(max(P-1,0))
   +K(a^2 U2(P)+2ab U2(max(P-1,0))+b^2 U2(max(P-2,0)))]/2,

DH=U2(P)/2+K[a U2(P)+b U2(max(P-1,0))]
   +a U3(P)+b U3(max(P-1,0))
   +K[a^2 U3(P)+2ab U3(max(P-1,0))+b^2 U3(max(P-2,0))].

Source Green upper tail:
 exp(-kappa D)[ exp(K a)(T1+T2)/2-DG ]/(4pi D).
Source gradient-component upper tail:
 exp(-kappa D)[ exp(K a)(K T1/2+(K+1)T2/2+T3)-DH ]/(4pi D^2).
Swap a,b and sum the omitted-degree union. For K=0 directly return U1(P)
and U2(P), avoiding cancellation. Omitted zero support is exact zero.

Native implementation: reuse the reviewed bounded direct positive tails and
geometric log sums. Evaluate the positive difference as
logX+log(-expm1(logY-logX)). Nonfinite, wrong-sign or unresolved near-equal
native representations mark the additional bound unavailable; they must
never create an admitted zero bound. Keep positive floors and all old
availability-safe minima. A strict new direct_first_wave_min_v1 identity
covers this changed bound; old rule outputs/configuration remain unchanged,
free-space only. All native p/x workspace caps remain as before.

Independent prospective90-digit total-degree/binomial coefficient sum at
the SHA-bound saved p12 gradient witness (D=.01913911745217461,
a=.15541472350336322,b=.09338003399270162,K=1.3614676064413669) gives
approximately G8.108469868062115e-9,H7.074440213522406e-6.
This finite reference is only a candidate diagnostic, not a full-system
upper enclosure or original admission. It motivates targeted native witness
and actual-kernel checks before any next original-size planner run. Existing
full-kernel/float32/64/RWG/current-grad/public/Bench/regression controls and
strict original growing-ka readmission are still required.

The saved-witness checks and general native controls pass. Terminal original-size,
public and cross-repository evidence is recorded in the phase assessment; no
production qualification is granted by this mathematical proof.
