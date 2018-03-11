import sys
sys.path.insert(0, '../qtrader/qtrader')

from AData import SHSZData

DATA_FOLRDER = '../data/SHSZ'

dl = SHSZData(DATA_FOLRDER)
dl.retry_d()
