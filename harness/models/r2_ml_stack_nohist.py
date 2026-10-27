"""r2_ml_stack sem NENHUM histórico (ablação: mede o valor das features da eleição anterior em 2022)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_ml_stack as S
class Model(S.Model):
    name = "r2_ml_stack_nohist"; USE_HIST = False
