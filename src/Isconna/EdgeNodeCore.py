from math import inf, log
from typing import Tuple

from numba import b1, f4, i4
from numba.core.types import string
from numba.experimental import jitclass
from numpy import ones, zeros
from numpy.random import randint

# region @jitclass
@jitclass({
	'bCur': b1[:],
	'bAcc': b1[:],
	'fCur': f4[:],
	'fAcc': f4[:],
	'wCur': f4[:],
	'wAcc': f4[:],
	'gCur': f4[:],
	'gAcc': f4[:],
	'wTime': i4[:],
	'gTime': i4[:],
})
# endregion
class CMSGroup:
	def __init__(self, length: int):
		self.bCur = zeros(length, b1)
		self.bAcc = zeros(length, b1)
		self.fCur = zeros(length, f4)
		self.fAcc = zeros(length, f4)
		self.wCur = zeros(length, f4)
		self.wAcc = zeros(length, f4)
		self.gCur = zeros(length, f4)
		self.gAcc = zeros(length, f4)
		self.wTime = ones(length, i4)
		self.gTime = ones(length, i4)

# region @jitclass
@jitclass({
	'nameAlg': string,
	'row': i4,
	'col': i4,
	'zeta': f4,
	'ts': f4,
	'param': i4[:],
	'edge': CMSGroup.class_type.instance_type,
	'source': CMSGroup.class_type.instance_type,
	'destination': CMSGroup.class_type.instance_type,
})
# endregion
class EdgeNodeCore:  # Cannot subclass from a jitclass
	def __init__(self, row: int, col: int, zeta: float = 0) -> None:
		self.nameAlg = 'Isconna-EN'
		self.row = row
		self.col = col
		self.zeta = zeta
		self.ts = 1
		self.param = randint(1, 1 << 16, 2 * row).astype(i4)
		self.edge = CMSGroup(row * col)
		self.source = CMSGroup(row * col)
		self.destination = CMSGroup(row * col)

	@staticmethod
	def GTest(c: float, a: float, t: float) -> float:
		return 0 if c == 0 or a == 0 or t <= 1 else 2 * c * abs(log(c * (t - 1) / a))

	def Reset(self, cms: CMSGroup) -> None:
		cms.fCur *= self.zeta
		for i in range(self.row * self.col):
			if not cms.bCur[i]:
				if cms.bAcc[i]:
					cms.gAcc[i] += cms.gCur[i]
					cms.gCur[i] *= self.zeta
					cms.gTime[i] += 1
				cms.gCur[i] += 1
		cms.bAcc, cms.bCur = cms.bCur, cms.bAcc
		cms.bCur.fill(False)

	def Update(self, a: int, b: int, cms: CMSGroup) -> Tuple[float, float, float]:
		fMinCur = fMinAcc = inf
		wMinTime = gMinTime = inf
		wIndex = gIndex = -1
		for i in range(self.row):
			i = i * self.col + ((a + 347 * b) * self.param[i] + self.param[i + self.row]) % self.col
			cms.fCur[i] += 1  # CMS Add
			cms.fAcc[i] += 1  # CMS Add
			fMinCur = min(fMinCur, cms.fCur[i])  # CMS Query
			fMinAcc = min(fMinAcc, cms.fAcc[i])  # CMS Query
			if not cms.bCur[i]:  # Haven't seen this edge in this timestamp
				if not cms.bAcc[i]:  # This edge didn't occur in last timestamp
					cms.wAcc[i] += cms.wCur[i]
					cms.wCur[i] *= self.zeta
					cms.wTime[i] += 1
				cms.wCur[i] += 1
				cms.bCur[i] = True  # Now it's seen
			if cms.wTime[i] < wMinTime:  # CMS ArgQuery
				wMinTime = cms.wTime[i]
				wIndex = i
			if cms.gTime[i] < gMinTime:  # CMS ArgQuery
				gMinTime = cms.gTime[i]
				gIndex = i
		return (
			self.GTest(fMinCur, fMinAcc, self.ts),
			self.GTest(cms.wCur[wIndex], cms.wAcc[wIndex], wMinTime),
			self.GTest(cms.gCur[gIndex], cms.gAcc[gIndex], gMinTime),
		)

	def Call(self, src: int, dst: int, ts: int, alpha: float, beta: float, gamma: float) -> float:
		if self.ts < ts:
			self.Reset(self.edge)
			self.Reset(self.source)
			self.Reset(self.destination)
			self.ts = ts
		efSc, ewSc, egSc = self.Update(src, dst, self.edge)
		sfSc, swSc, sgSc = self.Update(src, 000, self.source)
		dfSc, dwSc, dgSc = self.Update(dst, 000, self.destination)
		return max(efSc, sfSc, dfSc) ** alpha * max(ewSc, swSc, dwSc) ** beta * max(egSc, sgSc, dgSc) ** gamma
