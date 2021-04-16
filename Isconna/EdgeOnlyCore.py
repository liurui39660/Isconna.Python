from math import inf, log

from numba import bool_, float_, int_, types
from numba.experimental import jitclass
from numpy import ones, zeros
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
	"bCur": bool_[:],
	"bAcc": bool_[:],
	"fCur": float_[:],
	"fAcc": float_[:],
	"wCur": float_[:],
	"wAcc": float_[:],
	"gCur": float_[:],
	"gAcc": float_[:],
	"wTime": int_[:],
	"gTime": int_[:],
})
# endregion
class EdgeOnlyCore:
	def __init__(self, row: int, col: int, zeta: float = 0) -> None:
		self.nameAlg = "Isconna-EO"
		self.row = row
		self.col = col
		self.zeta = zeta
		self.ts = 1
		self.index = zeros(row, int_)
		self.param = randint(1, 1 << 16, 2 * row).astype(int_)

		self.bCur = zeros(row * col, bool_)
		self.bAcc = zeros(row * col, bool_)
		self.fCur = zeros(row * col, float_)
		self.fAcc = zeros(row * col, float_)
		self.wCur = zeros(row * col, float_)
		self.wAcc = zeros(row * col, float_)
		self.gCur = zeros(row * col, float_)
		self.gAcc = zeros(row * col, float_)
		self.wTime = ones(row * col, int_)
		self.gTime = ones(row * col, int_)

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

	# numba does not support __call__, so I use a sklearn-style name
	def FitPredict(self, src: int, dst: int, ts: int, alpha: float, beta: float, gamma: float) -> float:
		if self.ts < ts:
			self.fCur *= self.zeta
			for i in range(self.row * self.col):
				if not self.bCur[i]:
					if self.bAcc[i]:
						self.gAcc[i] += self.gCur[i]
						self.gCur[i] *= self.zeta
						self.gTime[i] += 1
					self.gCur[i] += 1
			self.bAcc, self.bCur = self.bCur, self.bAcc
			self.bCur.fill(False)
			self.ts = ts
		for i in range(self.row):
			self.index[i] = i = i * self.col + ((src + 347 * dst) * self.param[i] + self.param[i + self.row]) % self.col
			self.fCur[i] += 1
			self.fAcc[i] += 1
			if not self.bCur[i]:
				if not self.bAcc[i]:
					self.wAcc[i] += self.wCur[i]
					self.wCur[i] *= self.zeta
					self.wTime[i] += 1
				self.wCur[i] += 1
				self.bCur[i] = True
		wIndex = self.ArgQuery(self.wTime)
		gIndex = self.ArgQuery(self.gTime)
		fSc = self.GTest(self.Query(self.fCur), self.Query(self.fAcc), ts)
		wSc = self.GTest(self.wCur[wIndex], self.wAcc[wIndex], self.wTime[wIndex])
		gSc = self.GTest(self.gCur[gIndex], self.gAcc[gIndex], self.gTime[gIndex])
		return pow(fSc, alpha) * pow(wSc, beta) * pow(gSc, gamma)
