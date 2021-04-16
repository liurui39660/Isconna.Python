from math import inf, log
from typing import Tuple

from numba import bool_, float_, int_, types
from numba.experimental import jitclass
from numpy import ndarray, ones, zeros
from numpy.random import randint

# region @jitclass
@jitclass({
	"nameAlg": types.string,
	"row": int_,
	"col": int_,
	"zeta": float_,
	"ts": float_,
	"index": int_[:],
	"param": int_[:],

	"ebCur": bool_[:],
	"ebAcc": bool_[:],
	"efCur": float_[:],
	"efAcc": float_[:],
	"ewCur": float_[:],
	"ewAcc": float_[:],
	"egCur": float_[:],
	"egAcc": float_[:],
	"ewTime": int_[:],
	"egTime": int_[:],

	"sbCur": bool_[:],
	"sbAcc": bool_[:],
	"sfCur": float_[:],
	"sfAcc": float_[:],
	"swCur": float_[:],
	"swAcc": float_[:],
	"sgCur": float_[:],
	"sgAcc": float_[:],
	"swTime": int_[:],
	"sgTime": int_[:],

	"dbCur": bool_[:],
	"dbAcc": bool_[:],
	"dfCur": float_[:],
	"dfAcc": float_[:],
	"dwCur": float_[:],
	"dwAcc": float_[:],
	"dgCur": float_[:],
	"dgAcc": float_[:],
	"dwTime": int_[:],
	"dgTime": int_[:],
})
# endregion
class EdgeNodeCore:
	def __init__(self, row: int, col: int, zeta: float = 0) -> None:
		self.nameAlg = "Isconna-EN"
		self.row = row
		self.col = col
		self.zeta = zeta
		self.ts = 1
		self.index = zeros(row, int_)
		self.param = randint(1, 1 << 16, 2 * row).astype(int_)

		self.ebCur = zeros(row * col, bool_)
		self.ebAcc = zeros(row * col, bool_)
		self.efCur = zeros(row * col, float_)
		self.efAcc = zeros(row * col, float_)
		self.ewCur = zeros(row * col, float_)
		self.ewAcc = zeros(row * col, float_)
		self.egCur = zeros(row * col, float_)
		self.egAcc = zeros(row * col, float_)
		self.ewTime = ones(row * col, int_)
		self.egTime = ones(row * col, int_)

		self.sbCur = zeros(row * col, bool_)
		self.sbAcc = zeros(row * col, bool_)
		self.sfCur = zeros(row * col, float_)
		self.sfAcc = zeros(row * col, float_)
		self.swCur = zeros(row * col, float_)
		self.swAcc = zeros(row * col, float_)
		self.sgCur = zeros(row * col, float_)
		self.sgAcc = zeros(row * col, float_)
		self.swTime = ones(row * col, int_)
		self.sgTime = ones(row * col, int_)

		self.dbCur = zeros(row * col, bool_)
		self.dbAcc = zeros(row * col, bool_)
		self.dfCur = zeros(row * col, float_)
		self.dfAcc = zeros(row * col, float_)
		self.dwCur = zeros(row * col, float_)
		self.dwAcc = zeros(row * col, float_)
		self.dgCur = zeros(row * col, float_)
		self.dgAcc = zeros(row * col, float_)
		self.dwTime = ones(row * col, int_)
		self.dgTime = ones(row * col, int_)

	@staticmethod
	def GTest(c: float, a: float, t: float) -> float:
		return 0 if c == 0 or a == 0 or t <= 1 else 2 * c * abs(log(c * (t - 1) / a))

	def Query(self, data):
		least = inf
		for i in self.index:
			least = min(least, data[i])
		return least

	def ArgQuery(self, data) -> int:
		least = inf
		for i in self.index:
			if least > data[i]:
				least = data[i]
				arg = i
		return arg

	def ResetComponent(self, fCur: ndarray, bCur: ndarray, bAcc: ndarray, gCur: ndarray, gAcc: ndarray, gTime: ndarray) -> None:
		fCur *= self.zeta
		for i in range(self.row * self.col):
			if not bCur[i]:
				if bAcc[i]:
					gAcc[i] += gCur[i]
					gCur[i] *= self.zeta
					gTime[i] += 1
				gCur[i] += 1
		bAcc.fill(False)  # They swap outside, so this will become bCur

	def UpdateComponent(self, a: int, b: int, fCur: ndarray, fAcc: ndarray, bCur: ndarray, bAcc: ndarray, wCur: ndarray, wAcc: ndarray, wTime: ndarray, gCur: ndarray, gAcc: ndarray, gTime: ndarray) -> Tuple[float, float, float]:
		for i in range(self.row):
			self.index[i] = i = i * self.col + ((a + 347 * b) * self.param[i] + self.param[i + self.row]) % self.col
			fCur[i] += 1
			fAcc[i] += 1
			if not bCur[i]:
				if not bAcc[i]:
					wAcc[i] += wCur[i]
					wCur[i] *= self.zeta
					wTime[i] += 1
				wCur[i] += 1
				bCur[i] = True
		wIndex = self.ArgQuery(wTime)
		gIndex = self.ArgQuery(gTime)
		return (
			self.GTest(self.Query(fCur), self.Query(fAcc), self.ts),
			self.GTest(wCur[wIndex], wAcc[wIndex], wTime[wIndex]),
			self.GTest(gCur[gIndex], gAcc[gIndex], gTime[gIndex]),
		)

	def FitPredict(self, src: int, dst: int, ts: int, alpha: float, beta: float, gamma: float) -> float:
		if self.ts < ts:
			self.ResetComponent(self.efCur, self.ebCur, self.ebAcc, self.egCur, self.egAcc, self.egTime)
			self.ResetComponent(self.sfCur, self.sbCur, self.sbAcc, self.sgCur, self.sgAcc, self.sgTime)
			self.ResetComponent(self.dfCur, self.dbCur, self.dbAcc, self.dgCur, self.dgAcc, self.dgTime)
			self.ebAcc, self.ebCur = self.ebCur, self.ebAcc  # type: ndarray
			self.sbAcc, self.sbCur = self.sbCur, self.sbAcc  # type: ndarray
			self.dbAcc, self.dbCur = self.dbCur, self.dbAcc  # type: ndarray
			self.ts = ts
		efSc, ewSc, egSc = self.UpdateComponent(src, dst, self.efCur, self.efAcc, self.ebCur, self.ebAcc, self.ewCur, self.ewAcc, self.ewTime, self.egCur, self.egAcc, self.egTime)
		sfSc, swSc, sgSc = self.UpdateComponent(src, 000, self.sfCur, self.sfAcc, self.sbCur, self.sbAcc, self.swCur, self.swAcc, self.swTime, self.sgCur, self.sgAcc, self.sgTime)
		dfSc, dwSc, dgSc = self.UpdateComponent(dst, 000, self.dfCur, self.dfAcc, self.dbCur, self.dbAcc, self.dwCur, self.dwAcc, self.dwTime, self.dgCur, self.dgAcc, self.dgTime)
		return pow(max(efSc, sfSc, dfSc), alpha) * pow(max(ewSc, swSc, dwSc), beta) * pow(max(egSc, sgSc, dgSc), gamma)
