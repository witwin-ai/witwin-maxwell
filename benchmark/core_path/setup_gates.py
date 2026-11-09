"""Exact material, coefficient and table gates and synchronized prepare timings."""

from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack, contextmanager
import dataclasses
import gc
import importlib
import io
import json
from pathlib import Path
import time
import zlib
from unittest.mock import patch

import numpy as np
import torch
import witwin.maxwell as mw

from .gates import OUTPUT as GATE_OUTPUT, make_simulation, rule2_rows, tensors, tier_summary
from .scenes import CORE_SCENES, CoreCase, _grid_scene, _sphere, build_core_case
from witwin.maxwell.compiler import materials
from witwin.maxwell.compiler.accel import dispatch
from witwin.maxwell.scene import prepare_scene


OUTPUT = GATE_OUTPUT / "setup"
MODES = ((4, "kottke_edge"), (4, "kottke_smooth"), (3, "polarized"),
         (4, "polarized"), (1, "arithmetic"))
EXTRAS = ("vacuum64", "vacuum256", "odd", "graded", "custom", "periodic",
          "overlap", "rotated_box", "boolean", "sphere_box",
          "interface_pair", "conductive_kottke", "periodic_boolean", "empty_band")
O3B_SCENES = ("interface_pair", "conductive_kottke", "periodic_boolean", "empty_band")
O3C_SCENES = ("mesh_pyramid", "mesh_5k", "mesh_boolean")
EXTRAS += O3C_SCENES
S01_SCENES = ("cylinder_x", "cylinder_y", "cylinder_z", "cylinder_boolean",
              "pyramid_x", "pyramid_y", "pyramid_z", "polyslab", "polyslab_taper",
              "polyslab_concave", "pec_box", "pec_curved", "pec_conformal", "pmc",
              "periodic_cylinder", "periodic_polyslab", "conductive_periodic",
              "copper", "lumped_port", "thin_wire", "closed_surface", "dispersive", "thin_sheet")
EXTRAS += S01_SCENES
FLOAT64_SCENES = tuple("float64_" + kind for kind in (
    "box", "sphere", "cylinder_x", "cylinder_y", "cylinder_z", "pyramid", "polyslab",
    "union", "intersection", "difference", "smooth_union", "mixed_tree"))
F1_SCENES = ("f1_box_small", "f1_sphere_small", "f1_cylinder_small", "f1_mixed_medium")


# Frozen a302ac9 strict-xfail inputs: current torch output is the golden,
# including the known clamped-distance error. Keep original float64 vertices.
_SURFACE_TIE_INPUTS = (
    'c-qaIc~s5o`}em>^Qd`{=7FM=DCup8Oi{=b5u!mMDj^abrD-'
    '?qq1x>+Wy*MplMEdPImICynTd>XjG6tm@9A@%pKE<v>sinF=V`6hz25KZb#?9Q-'
    'h1Eo=l<+n&;&&#SJvM@)~po=4PSoi`cY;XuoB(+$1P9p!O~^z)%lxY#==P<fkCr+uo78o9T!EcTo~`@W98_(c!;BmmE+=N@vGv)<'
    'K``kUlj4(9}~WGWrW;c85zDjLav?My#~5ic{saRrC9wx{-'
    '{H?>B1R%HeQL1`)+tDfa1En`wv)%&~~&TGY~~c))3GBIa~mXXRiwJg^xFhhn;*{3gD6uv3nR~P+_Mlcx8&f&3}LA=razUo#|UM^P'
    '?2Kr<5D`Un#+oOST%4)Lb}@%OBd-CdC16oqyJZ64WO>Rtk5|0*|HlEnFx<^P=5ZF~9NgK3T=>&tyJ=KgvRDf+W~~(9y`MLImlAC%'
    'e2Za*<^}c5bvuIedMr`|2(b;gQSDM;?kLu+1LT^YE`Cu=}bmyHinuJM*`lTf46iR{NB#+K-5_u2<nXr>PutUb@_-'
    '5nhPKaeI6QdWg|tUK27ROoAJFeub5mTX8xrK4hMY1Q&wW$R2o#@L_YT&AfN{n7en6#`{BM*raZjJVsT3TYI(HP8P+ev$-'
    '{`tWbu^9{2l1sFb7CI^pAbJ{#5E6Mu{RNrw6n^S!VCEI{k2r`8J<3gM9d%csY4<vicKwMwG|KQ9`P<6)kSC!cDiQ@)nNZQ{`j2@l'
    'H<b|~jdN+KJrJBoBec9+9Umb_xrFh25+Y_*Kn%EGq%p;jsp5;WYNeQRtCAFETIA5<&jAf+PaVk}<@RnG=}&&ML9-'
    'e35sLrMO;eQs5Kv0RFe$8dU(Q#roGvDtRIMcAPCzI=|I0HaHmFYA#e!WVs&L#oLfB&4<Uo!3%^*t3Fpf6NkKYhrETxhy_LMXmAC%'
    'MxKo=O0VmFN%;AIU(*;0~d#8Pf63dT8>>pMe~6t_(=I`Zoe~x56P5Wk2qT-@M?{kd}4SB`s93!%)VTLMVGx7N9Ks-'
    '^QM+~QllJsQ{rx9-'
    'Qc6;L+QH}qlDO{(iHyuhyV&58tWfw6rrEvSjUf_g)myb%6yL(7jM#Tcs7+6<5HrNg_fBVqS|Yd{c5;)?CzgxW08;ZlZ$t@YD!VlI'
    'X&v&A};11uIXiC%SK4j_Thc4#i;vY8gk2!i=3!co1&L-;XgSe`6n+i0zXK<Hu1T*w_li2w=y4nRn1e3b_p>~L+@bCE-'
    '@~TUG6DTF2uBldI6;`B~U+o(r^7UF_PB(CX9H&LD42nv*+i;@Z1(((EeGB=ZZrHM!YMA=kyU_DW`<++1kQgds&Q~yH0#Ml$?ctDz'
    '*5Hks_=yJ!_Y+ME?4{x*#e|n2Vg|Y`)%RDMoSEo%5{}A@7v=uFjaP`1WPVym^zPSg$m2aQzek67?1Kze~<Qy8XI0dlrc0ug|wO4;'
    '2L{;^~C>jLAdAPS?Bnhs9W?<{J3o7cqp+fvi_=*wBkrGnrv3KzjSG8E<!%!?gP8;UuF%tTsF~{LqCmgx`3S78qBG`Yr7DI-'
    'd*CH?O6y?-m&@ywo2QCKh6_=79ahy*XG^;dX23LK!OTgGbaXEJI{<nP;0c9|~oI1@ls6aE-n_=yReF%|)w9wzKoFYtF3Vwjdcoms'
    '||;ep!mxMb|j37Ybl~Tw7N5vkX&;g0hU;1UR(rM52yzF<!GbH*5-!;e^-'
    'y*qy;8@GUwX@<}xxQC|Wa{v0ZUIjdwHTdf2p_v<J2Gb%##mBfT8A{h=1oId8-'
    '+ft1F=DWqJHXj*5JNAV0Wl(K0bNzg_6j%H<B&WCLK=HWl-pv8!SmRXL-d<9U9xhuOiU($)-'
    'Sp9jZ!8g#k~q#j!%MO1&(Zr>m3erxbL6Q*JOQEtW787;DnniD{jbj_72y8%9Uq%63!wdWrzIzWkMg1X;(G^jFeYJl#e+YE@X`$n$'
    ')CwbM9fdx7mf3gTcVd?7bn2Er}4vvD3-#Et1}`@KNow=M1iM;<<N18`@QpJ32wVs&1neBhFwN#b>tu^RO(K7de-pKr}}+O*tV_kY'
    '44d?nkGh3;Cro<<$Rc*;U3`qT7)<LcbnaN%22Shs!6?B1XbRiC7dBVj2V7=i@lP39r$wII_J6w@n_#kfBskuVUw$hvbPL<2aGBAR'
    'u&<KbL8zwdmiq4_kSJIB*osg>HANvD#MjNgD?2X*Js7M%_gQVB&eIdICsT)AuPtX&QbvvFJwiXb++>7>nPjEt4}!se_tuSs>(xt$'
    'Abf&A}O?Aoo;W{7QyoHFGE*r7T{US0ypPi0S-RNHe}5cp+sCaCtHmJ)~D-{X^{eGEZHZ}+9blU;G+?9J{RE2*!5?#gk^{^-'
    'tKY1REW4Ow^L+s#b~boyk^DiGJFwRF5JT_#qy^k9t<4M#mY6m#6<ZCa9-'
    'ALY`moqQ5VGK)x(Rx9UPe5DevcM(r&FH7Xb>q&~Hhod_6BP863O23^v2tV)pn8ar)M`=`)Y<vBzNVg>|uFOt_skntNV8KdDa(9ol'
    '%<`XGAdyaXZgtIYS;?Giy?Uc5-'
    'njDyurZszrPCcx0_z9r7T@G<^qv0#8HAE$OYZ~se8gxUQ+tNfu?f+x~nPUr`6VdCJQyS6}pJ(H~V#IG!c!$Z%EBa?Zsn6y$~bE^F'
    'Pu6t7CA1mM2g1qBPp762x?3-r^m12axK6|Y9v=S_{yI)<F#K*Il=c9({N?~jM<##39GQ5lbd02gP33~3dx^eQg7?-'
    'N_8mf*9F?uQ{+U_sGq*MiVjJ+7u*Hr8-'
    'Pb`C|V*R_=5hchKTRGHZOR!ntlQ_+~9O6rwm!@aPUyslH*X>mnBPh4u+njeoSdLJSy%SRc;XARqS}zGKTNj?vH7J9nPK1i379St9'
    '-tvA=E60<%3XjfUAui}By#Lv)1cQ%XKG|w7!m!>^*RRPRr@q?E3r$OS@E@f;^W|9qJ{V2c%FPlWK8+O}*Ruo}uBM&tN+O)C3EgYn'
    'EPq}OJ^#bekPnqd(H(`?%3-}Fr#QE=6n&QS#y86Qdc5KE2*qp>UO5yFv3M*1_wuVRjs1%7SH**<QJc##`o!a8wMZUPBm1f9-'
    'z&s?c8WUZPZ8!eTE2)&<v?jIOAzE%fETOW;;QV0xaAl!+vE)w_3ZuIw<vQ^S93<i(pLywyVLtudGlZ;F0f7s=iq9;B$lgCij2{lQ'
    'tiz7F#L4xM#_OA*slnPHBXRWUT(*>BMls|>z%f@&Mv}?)ZbcLTBQhn<DRrb%0sVX=UtqtxoE!|Va>iL!Do|8TZ@kIFzL3y*H@Db$'
    'BObR(rhWPMTuQm#lvyy>n6dW1?a5Xb=)sb{(e+GTsfP^LH*UrwsW8IvFhV{{kDrzY@hsgUDIkFR*f1IS?iID5lP=3u56TG>Ex&1*'
    '4ptf*Ld$_uh2rg4&EIobdn<7Shb_pkBiT1c;kKM=D>2}`&0hcL<pV3wR$J-'
    'cj7I<$+p&fw0dv6vMWyhz8GgVZ~ih42Ka6~>T1JAy6OC_75zjomIOEcd5;V2o4pg-'
    'U32iHwq?QFQ(`<wGOttn#09IlxF|)b0HJQf$K8>?&o?T12nJo`AoPxn)u?1Pj-NYV?bb(vvBqDQEqTJhhq;EUuXpe;B7TBk<a-'
    'f*^7VY>`I-'
    'mU>`8sU%8KDQw8ZUnvKSjA+z+o0a8TIxkY9R^i{Q}rw&o*6$h7o0#E#|R#PzpNVkA89Hw<YAy)VZ5D?aI>y*%u*8kctPS}{fhYZ{'
    's7iSWw_sX>7;7k5l%v}~^CATwd&!BBSz+<bSiB(u4wEqcsW-^YgejrGM>#Pan{;pn<iA4NFfv-'
    '`5zyaJrejO+NkMuc`h)#M}x3C66Rs#aZF0Q<LhdStqk<C#V~n<c;Q7(PGqV10W5UON6-'
    '@ou*i%TB7ZE=(0;Msv`PRc?jIj2;q{W+K62@r!BY0TQ(Qvet0$y&}XP7ufpGmBJx@ny^uRJvHn7fSe}_dH6lKzOcnmf~SY}|9&l6'
    'jI0^AI;2C2p?kt}bIe9DmZrw%*vqf4D?faEJ?k&|K7KdkkNA@ks13UEFp49_#P-'
    '<8qP0AH8+otO##s!T@h_hTZ589s$oAgq+QrB{o1iyjsRY5ci=Y2uDM7~cTL}}Cxk%sHU-'
    'j{+awImaQ%aLxUmF{U`Chd}c;%;7(z;QE+BC`Ps-'
    'Y5?&AoK)kWn$({MN=r`;=oum{RVt31ajv3jezEJr`A*YT{)MBFqcRY~m(}k!;ztAh@;=9l?5LuX2T$`tjrC<VW)B`jA-'
    '#{(KIcT@<_@{UO4Nj~fgm@^xr|Aa@!kJqM+;(sNCGgt+p>Vx32@2+teSQ?E_liU_w8v68W3#B`W^J~dZ_GxOaaT^^K$PotXe{(MX'
    'b)t9X1ZO_V)(J@ow*qK}?nFe`hXvxnx9}Ko=IEmq7_Aqe$dp36WJY8TcKksxNi~4qahe&?j_<ii&S6Q%~oW0MnU4r{LJq~xQ7opX'
    '5)7^(#^KkQFVc|6g8P?X{d_1H~2-'
    'Av2RhDKxvV|9>t4)zYYnAy*^B6JW1N%>JQO`!l%AvXYtECtoG&AC0eJS*{Ul&S$&BT76JH>uyCFpowRR70Q`Qw~5wjm)T2NQhy9~'
    'G+0;QqEUtM_IBE;zm0Gtg`+!kfz$wm+3%eu(5z-bnfWa>VrBv5#4}XTNdUxil$s1D4tJjufIHJ?C-'
    'J<V@V3T)ckjXA!0^e0D`)aw+<4S^f0P@eGu%IdFE@S^2!Xzgu;sp&YSRpOQQznUG)Bc`jHe!ZtO<PN(N(h;F!bB-'
    '|hq>e?%AEqN@)wUmp}-'
    'lnC9c6<;sDKHb`l?;M(Zi%4jQF5{5Q#s<YPM=?QECZ~y&rJ*$NigI=^!z*r0k&^<&B%<+$FIlC`vxb9(ONw4_H|eJxo1UF;mmQ_s'
    'LF`0zuYOtm9h614AmB+)589?!Xh@p)1$T>IVwNTJalf_d`%2{Ll3u>!TG4#zTN-JeEHwEcWuy=U!V6qsF^qTPBwzprtpqvh{5q*u'
    'qJ(g7+Yjc?pL?uf+wu^`m$So9!!i_t`#ib_kK9%Us|k<|9#G1{LeW*)$(tK8UOn^KkEPPobNHv?cbjByB;ztvuhkgw0+l@MdFOsB'
    '|PPr{H<MM3DFu|BbUS(?L>IWG4**}qZiRkK8M5^-IMT?vnTbkh;}0RSw!m+o^mWQUww&Y@&}VRqcsUnIbG7vP@<Xsy-'
    'A$WdW5GOQ{Sg+>_;?{KZL{?-'
    'GlIy(<Su=6Rk(`2NSJHc*?QJe2pfW$qyiLMjH~Ia+;){@kBHIk0o(N8xfvzO#KO6;~1iu{Bb1CXcfX!PLtFNAliuJ2M}#Yc*?QJe'
    '61#$$zMa_YyMelf8;4=Ncu@9n(2QNi8I=c@RVcfZ|NG7iDvRQlQ^T*2~RmgQg02>b|ild(bj~g9E;3XCeciOHi>7G^>KAKoHgmEo'
    'M@*1G7@LBGvO)6)EAL`LPYWfBrYKPkEk2Yn$*iC+L`2M6Kzj;%Jm@Y-'
    'aMk2^(~yl8Erv$%IT4HJC0~({a;GrjP65t$}#nqca8IjX7Xc6oY888r<@+C7fy5^k{?dA1>q^zlkEG$h-UV8K8Z8B7vU-AM6O#(i'
    'Dq7Z1tiXBN5WH%sbADJ`V!6L=aV?2l?YEcCsL13v?Iyq6WxpOl+z&ZyIi7~_m`N&8SOxL$~lvDLr65U{;VT$M%xgca!h@8*O)^zl'
    'fRb48LdEg$~lvIVxnzGzL;nS!c$J2>^mEXX7-m15@)n6;VEZF_RXJ&X7=Ae5@)nA;VH+|-`F*-'
    'Cz{EhNaBoECOqZrNWBcAjY)n6(YA!AoC>+_nng5oeHBXLj5a4c<&4O6V-V5I{xgZh8Ery%$}#olc8#-'
    '%X7VSKIHOewPdOt}FO+B#k{?R6IpHa%*flQf8r_NR-'
    '8Dv#IHPq4PdO%kQP&tlG_(Ijk~pI+2~RnuepJ^ukZ2}<35hQu>wi=?TyIh@f@n*UA3?MZ;VGv`=F636Cf|d^8LdTl%IT1PJc(xdA'
    '3)-'
    'a)+RjVnEGyAqYKeYet#10Pu738Za5uM&x2@flJ7yZ7U3zUNakw<(M<j*5@)ml;VGv@`Wa3%)4v~yGg_bUlw<1qca0;7X7WMeAnU('
    '>H=Gu!H;QO|l0S-'
    'Q1Hw~Ik<3>x(M<kS5@)n2;VEZ8`k78N)BiLQXS5mNDaX{G*)>ifn#m6#@es29&+LXXAoZpaZAS8^5^YL&$|;ihT1YgLznH`qll3v'
    'U8_tyUlRz}ne>{mZ+KTX$W9lc7eIkkEuO#u6WdBL(hBGDg787kn@)r}`oA8uVBKwXn(aio5P2$nye6ge(&XVk#D~MJj=gt*G_aQv'
    'xnEK1exo;WCUrORj$@z0xH=HG@7fo~@k{?ZUFTzt!f$TduL^JzK9*O6X^F{w|IBl|T780#Q&YgusI})C9O#NbV?kgtw1teZT&Y#8'
    'IaN49^9?^~@KaXe!!c$I}>^tj;X7-m<5>F-'
    'P3v|QjlYKLVXa#caOd;Ba@RVcfr;&4C8p&Tr;_JxyGp!p=pVUhw+J@w(5^YO(%BhllXExEy{t`ywVdQ)f(hX-'
    'u_RSeYE0c5Q45Ez*PdTRk9CGfPL-GSjJdm6}=XArFk$Pc78<YGnqRk0UIW@BH%qN=JUlK_?k=)-'
    '~*$rn!_RUp9tCDl)DxysYPdTRk0&?GV0m+|4;*-evdqFpx6{(j<v<b;iB-'
    '(=Tl>6~>>x(z%h?vi<)#TS+;j1H7{`1i_^U)<U{;RL3d$|0YkFEdn(Y1Qg1~r`&F6KNw**x1!1lfD@_6fQ4qxiYghxGg5EBEe9KW'
    '<%nFEjqDeg^&9PvWiApC#YMx1d~v`c9ivJ%x?RP*!MOLM0wg4pg4#l7s5|qpGhgXXBCcr{=Po+pvW9)X408HU@_!FE)M5MxUBfJ8'
    'ohd0;ZkQIes(;4JiTpWzu}i8CLDZuB$-'
    'p*B5JVyA~mL*!{pdZ!UKEEwvs~A%$hU;b(o{V$=<vaL!GMhtO=9s~=Aa<xi)dJP6@n*aHuiHJ^uq178<KUy$PYiW=FbQCysKJ!r0'
    'Zod+$RQAg5f8Dzf)RGSL3!8TK2Z*b1XQ}xQZfi9J}@oVf)yFO&$#!unKR|ge9DLUhBRr5Av{c8JZ?Z#}J%JFN?A6bZw5zqPamu|z'
    'Xnvxa6@(Qsu?EL1ct2_)FwQIA{bQuov&Q890qzFUTIF`N$;$!&m#f#J&WM~`m+tG<S9P}~oyxSPeM^Vv^H)d8ctUq4ABy)5Qk{wf'
    '35>xYW`H0)JMWvNEaz}Ssv3(9!a<1~SE*9XK-3*T_#g*{-)w@ShXAam?np8jZEP|1L@SLv^mFQnn@y2I-5kh-'
    'K$#ly27&%>W;larjuyZV{xiF;|Et%yO(hfeHK6%!7J(nTKBeit%0uEH2md`p~#mD+Fo=@HLWO!w|Xo|(rTr6pv{^rccd?<O{%Rl$'
    'H5(xpeBkY^<U|YMeE^=W3K1U3Z+1OTrKe)l=yo8NspF;u`t}R3ZOZk|aQ59S?e-}-'
    '9S&YKw_OHo$e020W^jrJ43T&KhzuR;h2bKImwyEp+7~*T>xM^(#)aSS8p0niQZR40R`Z0X$KK8(=?4}I2JZ{bEZNx^~=1WohA~tM'
    '?o3xe5AJ1<`*kfM|$VY^8eQBord7QN*-mKB43a)N7XKeinP~c)`@$Gd!RDa&#aKf+($CZxxO0&5Ldv?6ma4T2-'
    '{6*>|$sf<&1#aH&UU2bPBTrH6&jV*zkic2~cwYRpVZ_bLT&zexS=V-'
    'iho(mJHPQ0N!+Lioy=D#@3gtGscWd)tJ#XpMXC;*wz`6d$eosCcZ=VnpAI?L<fxT^CiYk${TBGH+8wEI^;r&1_gpJjEqksKtQ6($'
    '`4FX2@<6-uV#xkd;#W=Iise0R_3cQ`TtcU9|9{k(3O}W>}!M@Dg#bF&X^ct5In`prUdvcF!KJwT9*+<Ks9L|-Y@RR-'
    'a^=fRmZFaMMqn3xkyYJ1p+^-VHRMd0bP4iK5yrranYA$l#-_8G{X&Y)*vx3f87a((wUsi~19##tUA54kahH-'
    '&vCj*vq!8Y~{KjBym;hx`~s)oprepFC0q?L=^Edws}59Ofo+%W@pd-'
    '?ll(rLC|BNvq;GZy&o=HiZ5`)f%b84d_4mPD82B4qfcfJy0jQ1x{06C+;-'
    'PF3|7(2|vhm!gWCRbhEpSIe^AEMEs|ZLUiXu3;mm%sOVc1skXDgr!-@*Mae!ZN{FB#qf;XZ!=?mF<!sY&AcL$!a2)8_E$OwU-rwY'
    'gU@nsIOmr1K39sT)~n3lYI1Q{v@hZ16E61Ed)5q*uLD1>v(pD^)c)|5r+?Pj)UG|A8UNK+p1i&O^*Y;weDtkKwqjLsX{<neHR7w0'
    'TM3H9S0}zYIr=COUxWA><fzz__?pDmB$t-'
    'T#MdIe7P%BuCG(?Ad~MR78u4|AuS5D%C%!K6bxD63#MdLf9_de$_`QhVi}bBUe0}2Ulljpm^J74K1Ja)k@ePS@Ncz(yz7g?_NPl|'
    '7HzvL@>8}^@O^9zo`qn4DDe+Co{1}k=F(bYi>Cce(=EOHA{TUJ8g7_AsKV#zeCVp?yp9%3TiEl~zHYI)^;`br*V@Br3iuhKfKXc+'
    '+6W^NjXF+@$;@gn^dK2H4__m}!OXAxR-;VU%hxqoyw<q&sMdrtW_zt8$YvMZ+-'
    ';wlZLwqOVJCXiuiSJB&XVRY?@%s|LFX`K!`2C3AkIat)nI9M8yO91IiSJ5$SJIyo@%t0MKk3hz_ydSPfb`dw_-'
    '@2^BYpQHzB}>V$^5vG`RPIY9;824;wuoJ-Xf(R-6;}Zk@Po!_)6W}0(B#!64}??h;}DqPqMGO6Rq-JkNi1nW*nnOe)s=hkNokonr'
    '2^V=7Jf0Qqud+_mCgCCiGBnRs8Y)?7#mf{r6wx|MgNx{m*m!_kZ2fobTV+|NgD~@4r9hJd*#F#s4+u`~4DL*ZipckRy`68veO^Le'
    'swAllXqu;rs88rB=#c6o22LpgG^KufJbGfB*e4=ej4W>uQ_ke1C5L`@<{rWBs3JI%q;q<-cDj$}B(m3@!f8&E)?93||k>'
)


def tensor_metadata(value, path="model"):
    if torch.is_tensor(value):
        return {path: (str(value.dtype), str(value.device), tuple(value.shape))}
    if dataclasses.is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: metadata for name, item in value.items()
                for key, metadata in tensor_metadata(item, f"{path}/{name}").items()}
    if isinstance(value, (tuple, list)):
        return {key: metadata for index, item in enumerate(value)
                for key, metadata in tensor_metadata(item, f"{path}/{index}").items()}
    return {}


def snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    if dataclasses.is_dataclass(value):
        return {"_type": type(value).__name__,
                **{field.name: snapshot(getattr(value, field.name))
                   for field in dataclasses.fields(value)}}
    return value


@contextmanager
def record_setup():
    from witwin.maxwell.compiler.accel.build import build_extension

    class Calls(list):
        pieces = None

    calls = Calls()
    calls.pieces = {}
    original = dispatch.try_compile_material_model

    def record(*args, **kwargs):
        model = original(*args, **kwargs)
        calls.append(model is not dispatch.UNCHANGED)
        return model

    def record_piece(name, operation):
        def wrapped(*args, **kwargs):
            calls.pieces[name] = calls.pieces.get(name, 0) + 1
            return operation(*args, **kwargs)
        return wrapped

    extension = build_extension()
    with ExitStack() as stack:
        stack.enter_context(patch.object(dispatch, "try_compile_material_model", record))
        for name in ("sample", "poles", "distance", "coverage", "polygon_distance", "conductor_edges", "bulk", "window", "reconstruct", "coefficients", "hash", "gather", "remap",
                     "mesh_keys", "mesh_topology", "mesh_bounds", "mesh_distance"):
            stack.enter_context(patch.object(extension, name, record_piece(name, getattr(extension, name))))
        yield calls


def scene_for(name, *, device="cuda"):
    if name in FLOAT64_SCENES:
        return float64_scene(name, device=device)
    if name.startswith("f1_"):
        from benchmark.datagen.scenes import scene_builders

        builder = next(builder for builder in scene_builders() if builder.name == name)
        return builder.build(device=device).scene
    if name in S01_SCENES:
        return s01_scene(name, device=device)
    if name in CORE_SCENES:
        return build_core_case(name).scene.clone(device=device)
    from benchmark.scenes import SCENARIOS, build_scene
    if name in SCENARIOS:
        return build_scene(name).clone(device=device)
    if name.startswith("vacuum"):
        return _grid_scene((int(name[6:]),) * 3).clone(device=device)
    shape = ((7, 7, 7) if name == "mesh_5k" else (11, 11, 11)) if name in O3C_SCENES else (
        (17, 15, 19) if name == "odd" else (19, 19, 19))
    nodes = [np.linspace(-0.5, 0.5, size) for size in shape]
    if name == "graded":
        nodes = [np.sign(axis) * abs(axis) ** 1.3 for axis in nodes]
    scene = mw.Scene(
        domain=mw.Domain(bounds=tuple((float(axis[0]), float(axis[-1])) for axis in nodes)),
        grid=mw.GridSpec.custom(*nodes),
        boundary=mw.BoundarySpec.periodic() if name in ("periodic", "periodic_boolean") else mw.BoundarySpec.pec(),
        device=device,
    )
    sphere = mw.Sphere(position=(-0.46, 0, 0) if name in ("periodic", "periodic_boolean") else (0, 0, 0),
                       radius=0.23, device=device)
    box = mw.Box(position=(0.1, 0, 0), size=(0.35, 0.4, 0.3),
                 rotation=(0.17, 0.31, -0.2), device=device)
    if name in O3C_SCENES:
        geometry = mesh_geometry(large=name == "mesh_5k", device=device)
        if name == "mesh_boolean":
            geometry = mw.Difference(mw.Union(geometry, box, device=device),
                                     mw.Box(size=.15, device=device), smoothness=.017, device=device)
        scene.add_structure(mw.Structure(geometry=geometry, material=mw.Material(eps_r=3.7)))
    elif name == "empty_band":
        scene.add_structure(mw.Structure(
            geometry=mw.Box(size=4., device=device), material=mw.Material(eps_r=3.7)))
    elif name in ("boolean", "periodic_boolean"):
        geometry = mw.Union(sphere, box, smoothness=.017, device=device) if name == 'periodic_boolean' else mw.Intersection(
            mw.Union(sphere, box, device=device), mw.Box(size=0.48, device=device),
            smoothness=0.017, device=device,
        )
        scene.add_structure(mw.Structure(geometry=geometry, material=mw.Material(eps_r=3.7)))
    elif name == "rotated_box":
        scene.add_structure(mw.Structure(geometry=box, material=mw.Material(eps_r=3.7)))
    else:
        scene.add_structure(mw.Structure(
            geometry=sphere, material=mw.Material(eps_r=3.7, sigma_e=.013 if name == "conductive_kottke" else 0.),
            priority=2 if name == "overlap" else 0,
        ))
        if name in ("sphere_box", "overlap", "interface_pair", "conductive_kottke"):
            scene.add_structure(mw.Structure(
                geometry=box, material=mw.Material(eps_r=2.1), priority=1,
            ))
            scene.add_structure(mw.Structure(
                geometry=mw.Box(size=0.15, device=device),
                material=mw.Material(eps_r=5.3), priority=1,
            ))
    return scene


def float64_scene(name, *, device="cpu"):
    """Off-grid declarations; queries still follow the compiler's fp32 axes."""
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=device)

    transform = dict(position=tensor((.037, -.023, .019)), rotation=tensor((.17, .31, -.2)))
    box = mw.Box(size=tensor((.35, .4, .3)), **transform)
    sphere = mw.Sphere(position=tensor((-.071, .043, -.029)), radius=tensor(.213))
    kind = name.removeprefix("float64_")
    if kind == "box":
        geometry = box
    elif kind == "sphere":
        geometry = sphere
    elif kind.startswith("cylinder_"):
        geometry = mw.Cylinder(radius=tensor(.211), height=tensor(.473), axis=kind[-1], **transform)
    elif kind == "pyramid":
        geometry = mw.Pyramid(base_size=tensor(.37), height=tensor(.47), **transform)
    elif kind == "polyslab":
        geometry = mw.PolySlab(vertices=((- .2, -.17), (.23, -.13), (.19, .21), (-.16, .24)),
                               bounds=(-.21, .23), sidewall_angle=.13, **transform)
    else:
        cls = {"union": mw.Union, "intersection": mw.Intersection, "difference": mw.Difference,
               "smooth_union": mw.Union, "mixed_tree": mw.Difference}[kind]
        if kind == "mixed_tree":
            sphere = mw.Union(sphere, mw.Box(size=.19, device=device), device=device)
        geometry = cls(box, sphere, smoothness=tensor(.017) if kind == "smooth_union" else 0.,
                       position=tensor((.013, -.011, .007)), rotation=tensor((.03, -.02, .01)))
    scene = mw.Scene(domain=mw.Domain(bounds=((- .5, .5),) * 3),
                     grid=mw.GridSpec.uniform(.125), boundary=mw.BoundarySpec.pec(), device=device)
    scene.add_structure(mw.Structure(geometry, mw.Material(eps_r=3.7, mu_r=1.3, sigma_e=.013)))
    return scene


def s01_scene(name, *, device="cpu", n=19):
    small = name in ("copper", "lumped_port", "thin_wire", "closed_surface", "thin_sheet")
    bound = .02 if small else .5
    nodes = np.linspace(-bound, bound, n)
    periodic = name.startswith("periodic_") or name == "conductive_periodic"
    scene = mw.Scene(domain=mw.Domain(bounds=((-bound, bound),)*3),
                     grid=mw.GridSpec.custom(nodes, nodes, nodes),
                     boundary=(mw.BoundarySpec.periodic() if periodic else
                               mw.BoundarySpec.pmc() if name == "pmc" else mw.BoundarySpec.none()),
                     subpixel_samples=mw.SubpixelSpec(pec="conformal" if name == "pec_conformal" else "staircase"),
                     device=device)
    material = mw.Material(eps_r=3.7)
    position = (-.46, 0, 0) if periodic else (0, 0, 0)
    if name.startswith("cylinder_") or name == "periodic_cylinder":
        axis = name[-1] if name[-1] in "xyz" else "z"
        geometry = mw.Cylinder(position=position, radius=.21, height=.47, axis=axis,
                               rotation=(.17, .31, -.2), device=device)
        if name == "cylinder_boolean":
            geometry = mw.Difference(geometry, mw.Box(size=.18, device=device),
                                     smoothness=.017, device=device)
    elif name.startswith("pyramid_"):
        geometry = mw.Pyramid(position=(-.13, -.11, -.19), base_size=.4, height=.37,
                              axis=name[-1], rotation=(.17, .31, -.2), device=device)
    elif name.startswith("polyslab") or name == "periodic_polyslab":
        vertices = ((-.22, -.19), (.23, -.17), (.21, .2), (-.19, .21))
        if name == "polyslab_concave":
            vertices = ((-.22, -.2), (.23, -.2), (.23, .2), (0., .03), (-.22, .2))
        geometry = mw.PolySlab(vertices=vertices, bounds=(-.21, .23), position=position,
                              sidewall_angle=.13 if name == "polyslab_taper" else 0.,
                              rotation=(.17, .31, -.2), device=device)
    elif name.startswith("pec_") or name == "pmc":
        geometry = (mw.Sphere(radius=.23, device=device) if name == "pec_curved" else
                    mw.Box(position=(.031, -.017, .023), size=(.34, .31, .29), device=device))
        material = mw.Material.pmc() if name == "pmc" else mw.Material.pec()
    else:
        geometry = mw.Sphere(position=position, radius=.007 if small else .23, device=device)
        if name == "conductive_periodic":
            material = mw.Material(eps_r=3.7, sigma_e=.013)
        elif name == "dispersive":
            material = mw.Material(eps_r=2., lorentz_poles=(mw.LorentzPole(
                delta_eps=.7, resonance_frequency=1.5e9, gamma=.1e9),))
    scene.add_structure(mw.Structure(geometry=geometry, material=material, name="body"))
    if name == "copper":
        scene.add_structure(mw.Structure(name="foil", geometry=mw.Box(
            position=(float(nodes[3*n//4]), 0., 0.), size=(0., .02, .02), device=device),
            material=mw.CopperFoil(thickness=2.e-6, frequency_range=(.5e9, 2.e9), edge_model="none")))
    elif name == "thin_sheet":
        scene.add_structure(mw.Structure(name="sheet", geometry=mw.Box(
            position=(.01, 0., 0.), size=(0., .02, .02), device=device),
            material=mw.Medium2D(sigma_s=.01)))
    elif name == "lumped_port":
        k = n//2
        half = .5*(nodes[:-1]+nodes[1:])
        center = .5*(half[k-2]+half[k+1])
        extent = half[k+1]-half[k-2]
        scene.add_port(mw.LumpedPort("feed",
            positive=(float(nodes[k]), float(nodes[k]), float(nodes[k+1])),
            negative=(float(nodes[k]), float(nodes[k]), float(nodes[k-1])),
            voltage_path=mw.AxisPath("z"), current_surface=mw.Box(
                position=torch.tensor((center, center, half[k-1]), dtype=torch.float64, device=device),
                size=torch.tensor((extent, extent, 0.), dtype=torch.float64, device=device), device=device)))
    elif name == "thin_wire":
        scene.add_thin_wire(mw.ThinWire(name="wire", points=((.01, 0., -.01), (.01, 0., .01)),
            radius=1.e-4, conductor=mw.WireConductor.pec(),
            endpoints=(mw.WireEnd.open(), mw.WireEnd.open()), snap="continuous"))
    elif name == "closed_surface":
        scene.add_monitor(mw.ClosedSurfaceMonitor.box("surface", size=(.025,)*3, frequencies=(1.e9,)))
    return scene


def mesh_geometry(*, large=False, device="cpu"):
    if large:
        # Fifty latitude rings, fifty longitudes: exactly 5000 triangles.
        phi = np.arange(1, 51)[:, None] * np.pi / 51
        theta = np.arange(50)[None, :] * 2 * np.pi / 50
        vertices = np.stack(np.broadcast_arrays(np.sin(phi)*np.cos(theta),
                             np.sin(phi)*np.sin(theta), np.cos(phi)), -1).reshape(-1, 3)
        vertices = np.concatenate(([[0, 0, 1]], vertices, [[0, 0, -1]])) * .23
        faces = []
        for j in range(50):
            faces.append((0, 1+j, 1+(j+1)%50))
            for ring in range(49):
                a, b = 1+ring*50+j, 1+ring*50+(j+1)%50
                faces.extend(((a, a+50, b), (b, a+50, b+50)))
            faces.append((2501, 2451+(j+1)%50, 2451+j))
        vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
        faces = torch.as_tensor(faces, dtype=torch.int64, device=device)
    else:
        vertices, faces = mw.Pyramid(base_size=.4, height=.35, device=device).to_mesh()
    return mw.Mesh(vertices, faces, recenter=False, fill_mode="solid",
                   topology_diagnostics=False, device=device)


def mesh_tie_cases(*, device="cpu"):
    """CPU-constructible fixtures shared by the GPU gate and its dry run."""
    points = torch.tensor([[.25, .25, 0.], [.5, 0., 0.], [0., 0., 0.],
                           [.5, .5, 0.], [1., 1., 0.], [-0., +0., -0.],
                           [.25, .25, .5], [.25, .25, -.5], [1., 0., 0.],
                           [0., 1., 0.], [2., 2., 2.]], device=device)
    triangle = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], device=device)
    cases = [("face_edge_vertex_aabb", points, triangle),
             ("shared_coplanar", points, torch.cat((triangle, triangle.new_tensor([1., 1., 0.])-triangle))),
             ("duplicate_reversed_nonmanifold", points, torch.cat((triangle, triangle, triangle.flip(1)))),
             ("zero_area", points, torch.zeros_like(triangle)),
             ("sliver", points, triangle * triangle.new_tensor([1., 1.e-10, 1.])),
             ("equal_minima", points, torch.cat((triangle+triangle.new_tensor([0., 0., -.5]),
                                                  triangle+triangle.new_tensor([0., 0., .5]))))]
    threshold = triangle.new_tensor([[[1., 0., 0.], [-1., 1., 0.], [-1., -1., 0.]]])
    cases.extend((name, triangle.new_tensor([[0., 0., 0.], [-0., -0., -0.]]), t)
                 for name, t in (("angle_2pi_positive_zero", threshold),
                                 ("angle_2pi_negative_zero", threshold.flip(1))))
    generator = torch.Generator().manual_seed(302)
    random_points = torch.rand((67, 3), generator=generator).to(device)*3-1
    for n in (31, 32, 33, 255, 256, 257, 1023, 1024, 5000):
        translations = torch.rand((n, 1, 3), generator=generator).to(device)*4-2
        triangles = triangle.expand(n, -1, -1).clone() + translations
        # Repeated Morton keys, exact equal minima and degenerate leaves.
        triangles[:min(n, 16)] = triangle
        triangles[-1] = 0
        cases.append((f"tree_chunk_ties_{n}", torch.cat((points, random_points)), triangles))
    large = mesh_geometry(large=True, device=device)
    cases.append(("closed_5k", random_points, large.world_vertices[large.faces].contiguous()))
    fixture = np.load(io.BytesIO(zlib.decompress(base64.b85decode(_SURFACE_TIE_INPUTS))))
    for case in (0, 1):
        mesh = mw.Mesh(torch.as_tensor(fixture[f"v{case}"], device=device),
                       torch.as_tensor(fixture[f"f{case}"], device=device),
                       recenter=False, fill_mode="solid", topology_diagnostics=False, device=device)
        # Match the strict-xfail query's float32 world materialization exactly.
        triangles = mesh._world_vertices(device=torch.device(device), dtype=torch.float32)[mesh.faces].contiguous()
        cases.append((f"a302ac9_surface_current_output_{case}",
                      torch.tensor(fixture[f"axes{case}"], dtype=torch.float32, device=device)[None], triangles))
    return cases


def mesh_gate(output=OUTPUT):
    from witwin.core.geometry import mesh_sdf as core
    from witwin.maxwell.compiler.accel.mesh import accelerator

    rows = []
    assert not accelerator.supports(torch.zeros((1, 3)), torch.zeros((1, 3, 3)))
    for name, points, triangles in mesh_tie_cases(device="cuda"):
        assert not accelerator.supports(points, triangles)
        expected = (core._triangle_mesh_unsigned_distance_torch_from_triangles(points, triangles),
                    core._triangle_mesh_winding_angle_torch_from_triangles(points, triangles))
        signed = core._signed_distance_from_unsigned_and_winding(*expected, .5)
        if name.startswith("angle_2pi"):
            assert torch.equal(expected[1].abs(), torch.full_like(expected[1], 2*np.pi)), name
        cpu_bvh = core._build_triangle_bvh(triangles.cpu(), min_triangles=1)
        bvh = core._triangle_bvh_to_device(cpu_bvh, device=points.device)
        token = dispatch.setup_enabled.set(True)
        try:
            assert accelerator.supports(points, triangles)
            assert not accelerator.supports(points.requires_grad_(), triangles)
            points.requires_grad_(False)
            assert not accelerator.supports(points, triangles.detach().requires_grad_())
            assert not accelerator.supports(points.double(), triangles.double())
            with record_setup() as calls:
                for tree in (None, bvh):
                    # Exercise each hierarchy independently rather than reuse a
                    # cached device tree when testing the supplied static tree.
                    accelerator._cache.clear()
                    actual = accelerator.unsigned_distance_static_bvh(points, triangles, tree)
                    angle = accelerator.winding_angle(points, triangles)
                    actual_signed = accelerator.signed_distance_static_bvh(points, triangles, tree)
                    for label, a, b in (("distance", actual, expected[0]), ("winding", angle, expected[1]),
                                        ("signed", actual_signed, signed)):
                        assert torch.equal(a, b), (name, label, tree is not None, (a-b).abs().max().item())
                        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (name, label, "zero bits")
                vertices = triangles.reshape(-1, 3)
                faces = torch.arange(len(vertices), device=points.device).reshape(-1, 3)
                assert torch.equal(core.triangle_mesh_unsigned_distance(points, vertices, faces), expected[0])
                assert torch.equal(core.triangle_mesh_winding_angle(points, vertices, faces), expected[1])
                for beta in (.5, .17):
                    assert torch.equal(core.triangle_mesh_smooth_signed_distance(
                        points, vertices, faces, winding_beta=beta),
                        core._signed_distance_from_unsigned_and_winding(*expected, beta))
        finally:
            dispatch.setup_enabled.reset(token)
        row = dict(scene=name, points=len(points), triangles=len(triangles), accelerated=True,
                   accelerated_pieces=calls.pieces)
        if name.startswith("a302ac9"):
            row["current_output_golden"] = [v.cpu().tolist() for v in (*expected, signed)]
        rows.append(row)
        (output / "mesh-ties.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("MESH", row, flush=True)
    # Mutation and transform invalidation is owned by Mesh's query-data cache.
    mesh = mesh_geometry(large=True, device="cuda")
    queries = torch.tensor([[.03, .04, .05], [.3, .2, .1]], device="cuda")
    for mutation in (lambda: None, lambda: mesh.position.add_(.125), lambda: mesh.scale.mul_(.73),
                     lambda: mesh.rotation.add_(.07), lambda: mesh.vertices.add_(.003)):
        mutation()
        for mode in ("solid", "surface"):
            mesh.fill_mode = mode
            reference = mesh.signed_distance(*(queries[:, i:i+1, None] for i in range(3)))
            token = dispatch.setup_enabled.set(True)
            try:
                actual = mesh.signed_distance(*(queries[:, i:i+1, None] for i in range(3)))
            finally:
                dispatch.setup_enabled.reset(token)
            assert torch.equal(actual, reference), (mode, "mutation")
    for name, points, triangles in mesh_tie_cases(device="cuda")[:3]:
        # Topology diagnostics require host-authored vertices and faces; the mesh owns device placement.
        vertices = triangles.reshape(-1, 3).cpu().numpy()
        faces = torch.arange(len(vertices)).reshape(-1, 3).numpy()
        mesh = mw.Mesh(vertices, faces, recenter=False, fill_mode="auto",
                       topology_diagnostics=True, surface_thickness=.01, device="cuda")
        expected = mesh.signed_distance(*(points[:, i:i+1, None] for i in range(3)))
        token = dispatch.setup_enabled.set(True)
        try:
            actual = mesh.signed_distance(*(points[:, i:i+1, None] for i in range(3)))
        finally:
            dispatch.setup_enabled.reset(token)
        assert torch.equal(actual, expected), (name, "auto surface")
    # Reuse the same storage, then change its version, including strided inputs.
    triangles = mesh_geometry(large=True, device="cuda").world_vertices
    triangles = triangles[:2499].reshape(-1, 3, 3).transpose(1, 2)
    points = torch.tensor([[.1, .03], [.2, .04], [.3, .05]], device="cuda").T
    for translation in (0., .03125):
        triangles.add_(translation)
        expected = core._triangle_mesh_unsigned_distance_torch_from_triangles(points, triangles)
        token = dispatch.setup_enabled.set(True)
        try:
            actual = accelerator.unsigned_distance(points, triangles)
        finally:
            dispatch.setup_enabled.reset(token)
        assert torch.equal(actual, expected), "storage version/strides"
    return rows


def setup_rule2_rows(reference, candidate):
    """Retain exact model structure and nonnumeric metadata alongside tensor tiers."""
    def structure(left, right):
        assert type(left) is type(right), "Setup record type changed"
        if isinstance(left, dict):
            assert left.keys() == right.keys(), "Setup record keys changed"
            for key in left:
                structure(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right), "Setup record length changed"
            for a, b in zip(left, right):
                structure(a, b)
        elif not isinstance(left, (torch.Tensor, np.ndarray, np.number, bool, int, float, complex)):
            assert left == right, "Setup metadata changed"

    structure(reference, candidate)
    return rule2_rows(tensors(reference, "setup"), tensors(candidate, "setup"), level="setup")


def g4(names, output=OUTPUT, *, equivalence="bitwise"):
    from tests.fdtd.core_path.test_material_setup import assert_model_equal

    output = output / "g4.json"
    rows = json.loads(output.read_text()) if output.exists() else []
    completed = {(row["scene"], row["samples"], row["averaging"]) for row in rows
                 if row.get("equivalence", "bitwise") == equivalence and row.get("passed", True)}
    for name in names:
        for samples, averaging in MODES:
            if (name, samples, averaging) in completed:
                continue
            scene = scene_for(name)
            scene = scene.clone(subpixel_samples=mw.SubpixelSpec(
                samples=samples, averaging=averaging, pec=scene.subpixel.pec))
            start = time.perf_counter()
            print("G4 START", name, samples, averaging, flush=True)
            payloads, metadata = [], []
            for accelerated in (False, True):
                simulation = setup_simulation(scene, accelerated=accelerated)
                with record_setup() as calls:
                    prepared = simulation.prepare()
                solver = prepared.solver
                shape = (solver.Nx, solver.Ny, solver.Nz)
                payload = dict(model=solver._compiled_material_model, coefficients=coefficient_payload(solver))
                metadata.append(tensor_metadata(payload))
                payloads.append(snapshot(payload))
                assert not hasattr(solver, "_core_runtime")
                del payload, solver, prepared, simulation
                gc.collect()
                torch.cuda.empty_cache()
            assert metadata[1] == metadata[0]
            receipts = None
            if equivalence == "rule2":
                receipts = setup_rule2_rows(payloads[0], payloads[1])
            else:
                assert_model_equal(payloads[1], payloads[0])
            if name in O3C_SCENES:
                assert any(calls) and calls.pieces.get("mesh_distance"), name
            if name in (*S01_SCENES, *FLOAT64_SCENES) or name.startswith("f1_"):
                assert any(calls), (name, calls)
            row = dict(
                scene=name, samples=samples, averaging=averaging,
                accelerated=any(calls), accelerated_setup=calls, tensors=len(metadata[0]),
                accelerated_pieces=calls.pieces, piece_paths=piece_paths(calls),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                seconds=time.perf_counter() - start, shape=shape,
            )
            if receipts is not None:
                row.update(equivalence=equivalence, items=receipts, summary=tier_summary(receipts),
                           passed=all(item["tier"] != "T2" for item in receipts.values()))
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2) + "\n")
            if receipts is not None:
                write_setup_tensor_table(rows, output.with_name("g4-tensors.md"))
            if not row.get("passed", True):
                raise AssertionError(f"G4 numerical equivalence T2: {name}, {samples}, {averaging}")
            print("G4", row, flush=True)
            del payloads, scene
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def write_setup_tensor_table(rows, output):
    lines = ["| Scene | Samples | Mode | Tensor | Tier | Max normalized error | Bound ratio |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        for name, item in row.get("items", {}).items():
            lines.append(f"| {row['scene']} | {row['samples']} | {row['averaging']} | {name} | "
                         f"{item['tier']} | {item['max_rel']:.9g} | {item['ratio_to_bound']:.9g} |")
    output.write_text("\n".join(lines) + "\n")


def setup_simulation(scene, *, accelerated):
    return mw.Simulation.fdtd(scene, frequency=1.e9, accelerated_setup=accelerated, fdtd_core_path=False,
        run_time=mw.TimeConfig(time_steps=32), full_field_dft=False,
        spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False))


def coefficient_payload(solver):
    result = {key: value for key, value in vars(solver).items()
              if key.startswith(('cex_', 'cey_', 'cez_', 'chx_', 'chy_', 'chz_', 'cpml_'))
              and torch.is_tensor(value)}
    result['uniformity'] = solver._coefficient_uniformity
    return result


def piece_paths(calls):
    return dict(material_compile="accelerated" if any(calls) else "torch",
                sample_accumulation="accelerated" if calls.pieces.get("sample") else "torch",
                packed_distance_queries="accelerated" if calls.pieces.get("distance") else "torch",
                polygon_distance="accelerated" if calls.pieces.get("polygon_distance") else "torch",
                periodic_image_coverage="accelerated" if calls.pieces.get("coverage") else "torch",
                mesh_distance="accelerated" if calls.pieces.get("mesh_distance") else "torch",
                conductor_edge_fill="accelerated" if calls.pieces.get("conductor_edges") else "torch",
                binary_pole_edges="accelerated" if calls.pieces.get("poles") else "torch",
                coefficient_composition="accelerated" if calls.pieces.get("coefficients") else "torch",
                core_tables="accelerated" if calls.pieces.get("hash") else "not_requested",
                occupancy_tanh="torch", periodic_union_and_seams="torch")


def g1(names, output=OUTPUT):
    from tests.fdtd.core_path.test_material_setup import assert_model_equal

    output = output / "g1.json"
    rows = json.loads(output.read_text()) if output.exists() else []
    for name in names:
        if name in [row["scene"] for row in rows]:
            continue
        payloads, setups, metadata = [], [], []
        for arm in ("baseline", "candidate"):
            case = build_core_case(name)
            simulation = make_simulation(case, arm=arm, steps=64, full_field_dft=True)
            with record_setup() as calls:
                prepared = simulation.prepare()
            print("G1 SETUP", name, arm, calls, flush=True)
            if arm == "candidate":
                candidate_setup = calls
            setup = dict(model=prepared.solver._compiled_material_model,
                         uniformity=prepared.solver._coefficient_uniformity)
            metadata.append(tensor_metadata(setup, "setup"))
            setups.append(snapshot(setup))
            del setup
            if arm == "candidate":
                prepared.solver._core_runtime.chunk_size = 32
            result = prepared.run()
            torch.cuda.synchronize()
            payload = tensors({name: getattr(prepared.solver, name)
                               for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}, "fields")
            payload.update(tensors(result.fields, "phasors"))
            payload.update(tensors(result.monitors, "monitors"))
            payloads.append(payload)
            del result, prepared, simulation, case
            gc.collect()
            torch.cuda.empty_cache()
        assert metadata[1] == metadata[0], name
        assert_model_equal(setups[1], setups[0], "setup")
        assert_model_equal(payloads[1], payloads[0], name)
        row = dict(
            scene=name, tensors=len(payloads[0]), steps=64, K=32,
            accelerated=any(candidate_setup), accelerated_setup=candidate_setup,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        )
        rows.append(row)
        output.write_text(json.dumps(rows, indent=2) + "\n")
        print("G1", row, flush=True)
        del payloads, setups
    return rows


def timing(names=("vacuum", "sphere"), output=OUTPUT):
    from witwin.maxwell.compiler.accel.build import build_extension

    scene_module = importlib.import_module("witwin.maxwell.scene")
    build_extension()
    rows = []
    for name in names:
        for arm in ("baseline", "candidate"):
            scene = _grid_scene((256,) * 3)
            if name == "sphere":
                _sphere(scene)
            case = CoreCase(name, scene)
            simulation = make_simulation(case, arm=arm, steps=32, full_field_dft=False)
            stages = []
            original = materials.compile_material_model

            def record(*args, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model = original(*args, **kwargs)
                torch.cuda.synchronize()
                stages.append(time.perf_counter() - start)
                return model

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with patch.object(scene_module, "compile_material_model", record), record_setup() as calls:
                prepared = simulation.prepare()
            torch.cuda.synchronize()
            assert len(stages) == 1, stages
            row = dict(
                scene=name, arm=arm, prepare_seconds=time.perf_counter() - start,
                material_seconds=stages, accelerated=any(calls), accelerated_setup=calls,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
            )
            print("TIMING", row, flush=True)
            rows.append(row)
            (output / "timings.json").write_text(json.dumps(rows, indent=2) + "\n")
            del prepared, simulation, case, scene
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def coefficient_gate(names, output=OUTPUT):
    from tests.fdtd.core_path.test_material_setup import assert_model_equal

    rows = []
    configurations = [(name, policy) for name in names
                      for policy in (MODES if name in S01_SCENES else MODES[:2] if name in O3B_SCENES else (None,))]
    for name, policy in configurations:
        payloads = []
        cases = [build_core_case(name) if name in CORE_SCENES
                 else CoreCase(name, scene_for(name))]
        if policy is not None:
            samples, averaging = policy
            cases = [CoreCase(name, cases[0].scene.clone(
                subpixel_samples=mw.SubpixelSpec(samples=samples, averaging=averaging, pec=cases[0].scene.subpixel.pec)))]
        for arm in ('baseline', 'candidate'):
            simulation = (setup_simulation(cases[0].scene, accelerated=arm == 'candidate')
                          if name in S01_SCENES else
                          make_simulation(cases[0], arm=arm, steps=32, full_field_dft=False))
            with record_setup() as calls:
                prepared = simulation.prepare()
            solver = prepared.solver
            coefficients = {key: value for key, value in vars(solver).items()
                            if key.startswith(('cex_', 'cey_', 'cez_', 'chx_', 'chy_', 'chz_', 'cpml_'))
                            and torch.is_tensor(value)}
            coefficients['uniformity'] = solver._coefficient_uniformity
            payloads.append(snapshot(coefficients))
            if arm == 'candidate':
                pieces = calls.pieces
            del solver, prepared, simulation
            gc.collect()
            torch.cuda.empty_cache()
        assert_model_equal(payloads[1], payloads[0], name)
        row = dict(scene=name, policy=policy, tensors=len(tensor_metadata(payloads[0])),
                   accelerated_pieces=pieces, accelerated=bool(pieces.get('coefficients')))
        rows.append(row)
        (output / 'coefficients.json').write_text(json.dumps(rows, indent=2) + '\n')
        print('COEFFICIENTS', row, flush=True)
    return rows


def primitive_gate(output=OUTPUT):
    from witwin.maxwell.compiler.accel.primitives import sampling_geometry
    from witwin.maxwell.compiler.accel.geometry import native_geometry
    from witwin.maxwell.compiler.subpixel import _subsample_fill_fraction

    rows = []
    generator = torch.Generator(device="cpu").manual_seed(13)
    points = (torch.rand((10003, 3), generator=generator)*1.4-.7).cuda()
    points = torch.cat((points, points.new_tensor([[0., -0., 0.], [.23, 0., 0.], [-.5, .5, 0.]])))
    coords = tuple(points[:, i:i+1, None] for i in range(3))
    for name in S01_SCENES[:10]:
        geometry = prepare_scene(s01_scene(name, device="cuda")).structures[0].geometry
        payloads = []
        for accelerated in (False, True):
            token = dispatch.setup_enabled.set(accelerated)
            try:
                sample = sampling_geometry(geometry) if accelerated else geometry
                with record_setup() as calls:
                    payloads.append((sample.signed_distance(*coords),
                        _subsample_fill_fraction(sample, *coords, (.051, .063, .047)),
                        sample.to_mask(*coords, beta=.002)))
                    if accelerated and native_geometry(geometry):
                        direct = sampling_geometry(geometry, spacing=(.051, .063, .047)).to_mask(*coords)
                        assert torch.equal(direct, payloads[0][1]), name
                        assert torch.equal(direct.view(torch.int32), payloads[0][1].view(torch.int32)), name
            finally:
                dispatch.setup_enabled.reset(token)
        for actual, expected in zip(payloads[1], payloads[0]):
            assert torch.equal(actual, expected), name
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), name
        row = dict(scene=name, queries=len(points), accelerated=bool(calls.pieces), accelerated_pieces=calls.pieces)
        rows.append(row)
        print("PRIMITIVE", row, flush=True)
    (output / "primitives.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def s01_timing(names=S01_SCENES, output=OUTPUT):
    from witwin.maxwell.compiler.accel.build import build_extension

    build_extension()
    rows = []
    for name in names:
        for samples, averaging in MODES:
            for accelerated in (False, True):
                scene = s01_scene(name, device="cuda", n=256)
                scene = scene.clone(subpixel_samples=mw.SubpixelSpec(
                    samples=samples, averaging=averaging, pec=scene.subpixel.pec))
                simulation = setup_simulation(scene, accelerated=accelerated)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with record_setup() as calls:
                    prepared = simulation.prepare()
                torch.cuda.synchronize()
                solver = prepared.solver
                assert (solver.Nx, solver.Ny, solver.Nz) == (256,)*3
                row = dict(scene=name, samples=samples, averaging=averaging, shape=(256,)*3,
                    arm="accelerated" if accelerated else "torch", prepare_seconds=time.perf_counter()-start,
                    accelerated=any(calls), accelerated_pieces=calls.pieces, piece_paths=piece_paths(calls),
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
                rows.append(row)
                print("S01 TIMING", row, flush=True)
                (output / "s01-timings.json").write_text(json.dumps(rows, indent=2) + "\n")
                del solver, prepared, simulation, scene
    return rows


def table_gate(names, output=OUTPUT):
    from tests.fdtd.core_path.test_material_setup import assert_model_equal
    from witwin.maxwell.compiler.accel import tables
    from witwin.maxwell.fdtd.runtime.core_path import _compress_coefficients

    rows = []
    for name in names:
        if name == 'sphere256_edge4':
            scene = _sphere(_grid_scene((256,) * 3)).clone(
                subpixel_samples=mw.SubpixelSpec(samples=4, averaging='kottke_edge'))
            case = CoreCase(name, scene)
        else:
            case = build_core_case(name)
        # Isolate compression from stepping and setup acceleration.
        solver = make_simulation(case, arm='baseline', steps=32, full_field_dft=False).prepare().solver
        for family in ('e', 'h'):
            torch.cuda.reset_peak_memory_stats()
            with patch.object(tables, 'compress', return_value=dispatch.UNCHANGED):
                expected = _compress_coefficients(solver, family)
            metadata = tensor_metadata(expected)
            baseline = snapshot(expected)
            del expected
            torch.cuda.synchronize()
            start = time.perf_counter()
            with record_setup() as calls:
                actual = _compress_coefficients(solver, family)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            assert tensor_metadata(actual) == metadata
            actual = snapshot(actual)
            assert_model_equal(actual, baseline, name + '/' + family)
            # Compare signed-zero bit patterns in the float table as well.
            assert torch.equal(actual[1].view(torch.int32), baseline[1].view(torch.int32))
            row = dict(scene=name, family=family, rows=actual[1].shape[0] if actual[1].ndim else 0,
                       uniform=actual[3], accelerated=bool(calls.pieces.get('hash')),
                       accelerated_pieces=calls.pieces, seconds=seconds,
                       peak_allocated=torch.cuda.max_memory_allocated(),
                       peak_reserved=torch.cuda.max_memory_reserved())
            rows.append(row)
            (output / 'tables.json').write_text(json.dumps(rows, indent=2) + '\n')
            print('TABLES', row, flush=True)
            del actual, baseline
        del solver, case
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def window_gate(output=OUTPUT):
    from witwin.maxwell.compiler.accel.build import build_extension
    from witwin.maxwell.compiler.subpixel import _kottke_edge_diagonal

    rows = []
    extension = build_extension()
    for samples in (27, 64):
        for count in (1, 2, 3, 7, 15, 16, 17, 31, 64, 4095, 4096):
            generator = torch.Generator(device='cuda').manual_seed(count)
            sample = torch.rand((count, samples), device='cuda', generator=generator)*5+1
            offsets = torch.randn((count, samples, 3), device='cuda', generator=generator)*.1
            for axis in range(3):
                expected = _kottke_edge_diagonal(sample, offsets, axis)
                actual = extension.window(sample, offsets, axis)
                assert torch.equal(actual, expected), (samples, count, axis)
            row = dict(samples=samples, windows=count, axes=3, accelerated=True)
            rows.append(row)
            print('WINDOWS', row, flush=True)
    (output / 'windows.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def o3b_timing(names, output=OUTPUT):
    from witwin.maxwell.fdtd.runtime import core_path
    from witwin.maxwell.compiler.accel.build import build_extension
    from witwin.maxwell.fdtd.cuda.build import build_extension as build_runtime
    from witwin.maxwell.fdtd.cuda.core.build import build_extension as build_core

    build_extension()
    build_runtime()
    build_core()
    rows = []
    modes = {'edge4': (4, 'kottke_edge'), 'smooth4': (4, 'kottke_smooth'), 'polarized3': (3, 'polarized')}
    for name in names:
        grid, mode = name.split('_')
        n = int(grid.removeprefix('pair'))
        samples, averaging = modes[mode]
        for arm in ('baseline', 'candidate'):
            scene = _sphere(_grid_scene((n,) * 3)).clone(subpixel_samples=mw.SubpixelSpec(
                samples=samples, averaging=averaging))
            scene.add_structure(mw.Structure(
                geometry=mw.Box(position=(.24, 0, 0), size=(.12, .24, .2), device='cpu'),
                material=mw.Material(eps_r=3.)))
            simulation = make_simulation(CoreCase(name, scene), arm=arm, steps=32, full_field_dft=False)
            table_seconds = []
            original = core_path._compress_coefficients

            def timed_table(*args):
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = original(*args)
                torch.cuda.synchronize()
                table_seconds.append(time.perf_counter() - start)
                return value

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with patch.object(core_path, '_compress_coefficients', timed_table), record_setup() as calls:
                prepared = simulation.prepare()
            torch.cuda.synchronize()
            row = dict(scene=name, arm=arm, shape=(n, n, n), prepare_seconds=time.perf_counter()-start,
                       table_seconds=table_seconds, accelerated=any(calls), accelerated_pieces=calls.pieces,
                       peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            rows.append(row)
            print('TIMING', row, flush=True)
            (output / 'o3b-timings.json').write_text(json.dumps(rows, indent=2) + '\n')
            del prepared, simulation, scene
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def o3c_timing(names=("mesh128", "mesh256"), output=OUTPUT):
    from witwin.core.geometry import mesh_sdf as core
    from witwin.maxwell.compiler.accel.mesh import accelerator
    from witwin.maxwell.compiler.accel.build import build_extension

    build_extension()
    rows = []
    for name in names:
        n = int(name.removeprefix("mesh"))
        for arm in ("baseline", "candidate"):
            scene = _grid_scene((n,) * 3, subpixel=mw.SubpixelSpec(samples=1, averaging="arithmetic"))
            scene.add_structure(mw.Structure(geometry=mesh_geometry(large=True), material=mw.Material(eps_r=3.7)))
            simulation = make_simulation(CoreCase(name, scene), arm=arm, steps=32, full_field_dft=False)
            split = {"distance": [], "winding": []}

            def timed(label, operation):
                def run(*args, **kwargs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    value = operation(*args, **kwargs)
                    torch.cuda.synchronize()
                    split[label].append(time.perf_counter()-start)
                    return value
                return run

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with ExitStack() as stack:
                stack.enter_context(patch.object(core, "_triangle_mesh_unsigned_distance_torch_from_triangles",
                    timed("distance", core._triangle_mesh_unsigned_distance_torch_from_triangles)))
                stack.enter_context(patch.object(core, "_triangle_mesh_winding_angle_torch_from_triangles",
                    timed("winding", core._triangle_mesh_winding_angle_torch_from_triangles)))
                stack.enter_context(patch.object(accelerator, "unsigned_distance_static_bvh",
                    timed("distance", accelerator.unsigned_distance_static_bvh)))
                with record_setup() as calls:
                    prepared = simulation.prepare()
            torch.cuda.synchronize()
            row = dict(scene=name, arm=arm, shape=(n, n, n), triangles=5000,
                       samples=1, averaging="arithmetic", prepare_seconds=time.perf_counter()-start,
                       prepare_distance_seconds=split["distance"], prepare_winding_seconds=split["winding"],
                       accelerated=any(calls), accelerated_pieces=calls.pieces,
                       peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            del prepared, simulation, scene
            gc.collect()
            torch.cuda.empty_cache()
            # Isolate full-grid distance from exact winding; GPU-authored data
            # exercises LBVH creation, followed by a warm cached distance query.
            geometry = mesh_geometry(large=True, device="cuda")
            triangles = geometry.world_vertices[geometry.faces].contiguous()
            axis = torch.linspace(-.64, .64, n, device="cuda")
            points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
            token = dispatch.setup_enabled.set(arm == "candidate")
            try:
                vertices, faces = triangles.reshape(-1, 3), torch.arange(15000, device="cuda").reshape(-1, 3)
                for label, operation in (("distance_cold", core.triangle_mesh_unsigned_distance),
                                         ("distance_warm", core.triangle_mesh_unsigned_distance),
                                         ("winding", core.triangle_mesh_winding_angle)):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    value = operation(points, vertices, faces, _triangles=triangles)
                    torch.cuda.synchronize()
                    row[f"raw_{label}_seconds"] = time.perf_counter()-start
                    del value
            finally:
                dispatch.setup_enabled.reset(token)
            rows.append(row)
            (output / "o3c-timings.json").write_text(json.dumps(rows, indent=2) + "\n")
            print("O3C TIMING", row, flush=True)
            del points, triangles, geometry, vertices, faces
            accelerator._cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def cpu_dry_run(output=OUTPUT):
    from witwin.maxwell.compiler.accel.geometry import native_geometry, qualified_geometry
    from witwin.maxwell.compiler.accel.mesh import accelerator

    rows = []
    for name in (*EXTRAS, *CORE_SCENES):
        for samples, averaging in MODES:
            scene = scene_for(name, device="cpu")
            scene = scene.clone(subpixel_samples=mw.SubpixelSpec(
                samples=samples, averaging=averaging, pec=scene.subpixel.pec))
            simulation = setup_simulation(scene, accelerated=True)
            assert simulation.config.accelerated_setup and not simulation.config.fdtd_core_path
            prepared_scene = prepare_scene(scene)
            assert torch.device(prepared_scene.device).type == "cpu"
            assert not dispatch.eligible(prepared_scene, 1., 1., prepared_scene.subpixel)
            rows.append(dict(scene=name, samples=samples, averaging=averaging, accelerated=False))
    for name in O3C_SCENES:
        scene = scene_for(name, device="cpu")
        geometry = scene.structures[0].geometry
        assert qualified_geometry(geometry) and not native_geometry(geometry)
        assert dispatch._fp32_geometry(geometry)
        assert not dispatch.requires_grad(geometry)
        if name != "mesh_boolean":
            assert geometry.face_count == (5000 if name == "mesh_5k" else 6)
        rows.append(dict(scene=name, geometry=type(geometry).__name__, accelerated=False))
    for name, points, triangles in mesh_tie_cases():
        assert points.device.type == triangles.device.type == "cpu"
        assert not accelerator.supports(points, triangles)
        rows.append(dict(scene=name, points=len(points), triangles=len(triangles), accelerated=False))
    from witwin.maxwell.compiler.ports import compile_port_geometry
    from witwin.maxwell.compiler.copper_foil import compile_copper_faces
    for n in (19, 256):
        port_scene = s01_scene("lumped_port", n=n)
        compile_port_geometry(port_scene, port_scene.ports[0], device="cpu")
        faces = compile_copper_faces(prepare_scene(s01_scene("copper", n=n)), device="cpu")
        assert len(faces) == 2
    assert not torch.cuda.is_initialized()
    (output / "cpu-dry-run.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("CPU DRY RUN", len(rows), "fixtures; CUDA initialized:", torch.cuda.is_initialized(), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=("g4", "g1", "coefficients", "tables", "windows", "timing", "o3b-timing",
                                         "mesh-ties", "o3c-timing", "cpu-dry-run", "primitives", "s01-timing"))
    parser.add_argument("names", nargs="*", help="scene names (default: the complete gate scene list)")
    parser.add_argument("--output", type=Path, default=OUTPUT, help=f"receipt directory (default: {OUTPUT})")
    parser.add_argument("--equivalence", choices=("bitwise", "rule2"), default="bitwise")
    args = parser.parse_args()
    if args.equivalence != "bitwise" and args.gate != "g4":
        parser.error("--equivalence rule2 is supported only for g4")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.gate == "cpu-dry-run":
        cpu_dry_run(args.output)
        return
    if args.gate == "mesh-ties":
        rows = mesh_gate(args.output)
    elif args.gate == "o3c-timing":
        rows = o3c_timing(args.names or ("mesh128", "mesh256"), args.output)
    elif args.gate == "g4":
        rows = g4(args.names or (*EXTRAS, *CORE_SCENES), args.output, equivalence=args.equivalence)
    elif args.gate == "g1":
        rows = g1(args.names or CORE_SCENES, args.output)
    elif args.gate == "coefficients":
        rows = coefficient_gate(args.names or (*CORE_SCENES, *O3B_SCENES, *S01_SCENES), args.output)
    elif args.gate == "primitives":
        rows = primitive_gate(args.output)
    elif args.gate == "s01-timing":
        rows = s01_timing(args.names or S01_SCENES, args.output)
    elif args.gate == "tables":
        rows = table_gate(args.names or (*CORE_SCENES, 'sphere256_edge4'), args.output)
    elif args.gate == "windows":
        rows = window_gate(args.output)
    elif args.gate == "o3b-timing":
        rows = o3b_timing(args.names or tuple(f'pair{n}_{mode}' for n in (256, 384)
                                            for mode in ('edge4', 'smooth4', 'polarized3')), args.output)
    else:
        rows = timing(args.names or ("vacuum", "sphere"), args.output)
    accelerated = sum(row["accelerated"] for row in rows)
    print("SETUPS", dict(accelerated=accelerated, torch=len(rows) - accelerated), flush=True)
    print("PEAK", torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(), flush=True)


if __name__ == "__main__":
    main()
