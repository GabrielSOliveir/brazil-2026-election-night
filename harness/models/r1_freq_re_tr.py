"""r1_freq_re_tr — igual ao r1_freq_re, mas f(.) inclui as fatias de transferência da zona
(Tebet, Ciro, outros, brancos+nulos, abstenção) além de b1 e b1^2. Testa se a composição do 1T acrescenta algo
depois de controlar b1 e geografia (região/UF/município)."""
import sys; sys.path.insert(0,"harness/models")
from r1_freq_re import Model as Base
class Model(Base):
    name="r1_freq_re_tr"; kind="full"
