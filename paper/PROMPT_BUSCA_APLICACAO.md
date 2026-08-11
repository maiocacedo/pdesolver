# Prompt para busca de aplicação física — outro chat

Copie tudo abaixo da linha para um chat novo (de preferência com busca web ativa).

---

Preciso da sua ajuda para encontrar **um problema de física real** que sirva como
estudo de aplicação em um artigo de software submetido ao *Computer Physics
Communications* (CPC). O software é um solver de EDPs por diferenças finitas, e
o artigo hoje só tem problemas de verificação de livro-texto — falta uma
aplicação que produza um **resultado físico**, não apenas confirmação de
convergência. Esse é o principal motivo pelo qual o artigo ainda não está pronto
para a CPC.

## O que o software resolve (envelope exato — respeite-o)

Sistemas acoplados de EDPs **de primeira ordem no tempo**:

    ∂u_i/∂t = F_i(u_1, ..., u_N, ∇u_i, ∇²u_i, ∂²u_i/∂x∂y, x, t)

- **Domínios:** retangulares cartesianos, **1D e 2D apenas**
- **Malhas:** uniformes ou estiradas (Chebyshev, tanh, uni/bilateral), estruturadas
- **Contornos:** Dirichlet, Neumann, Robin e periódicos, independentes por face,
  com expressões simbólicas em x, y, t
- **Termos:** não-linearidades arbitrárias, derivadas mistas de 2ª ordem,
  coeficientes variáveis no espaço e no tempo
- **Heterogeneidade:** equações distintas por região (obstáculos internos,
  interfaces entre materiais com forma conservativa de fluxo)
- **Integradores:** BDF2, Crank–Nicolson, Runge–Kutta–Fehlberg adaptativo
- **Escala:** malhas 2D até ~10⁵ incógnitas; GPU acima de ~65 000 incógnitas
- **Precisão espacial:** segunda ordem

Equações hiperbólicas de 2ª ordem servem se reescritas como sistema de 1ª ordem
(ex.: onda → ∂ₜu = v, ∂ₜv = c²∂ₓₓu).

## O que ele NÃO resolve (descarte candidatos que exijam isso)

- **3D** ou geometria não estruturada / curvilínea / com fronteira móvel
- Problemas **puramente elípticos** (sem derivada temporal)
- **Choques** em leis de conservação — não há limitadores nem esquemas TVD/WENO
- Precisão **acima de 2ª ordem** ou acurácia espectral
- Variáveis **complexas** (ex.: Schrödinger não-linear na forma usual)
- Ordem espacial **acima de 2** (ex.: Cahn–Hilliard, que é de 4ª ordem)
- Acoplamento com malha adaptativa, multigrid ou decomposição de domínio

## O que eu quero de você

Proponha **3 a 5 candidatos**, ordenados pela sua avaliação de força. Para cada um:

1. **Problema e equações**, escritas explicitamente na forma acima. Confirme que
   cabem no envelope, item por item.
2. **Por que é física de verdade** — que quantidade de interesse o cálculo
   produz? (padrão, frente, tempo característico, diagrama de fase, coeficiente
   efetivo). Não aceito "é um caso de teste conhecido".
3. **Referência de validação**: solução analítica, resultado experimental ou
   resultado numérico publicado contra o qual comparar. Cite o trabalho.
4. **Ligação com a CPC**: encontre artigos *publicados na CPC* (ou em veículos
   equivalentes: Journal of Computational Physics, SoftwareX, Physical Review E)
   sobre esse problema. Isso mostra que o tema interessa ao público do periódico.
   **Dê DOI ou link de cada um.**
5. **O que o software acrescentaria**, se é que acrescenta. Seja cético: se o
   problema já é bem resolvido por ferramenta existente (FiPy, py-pde, Dedalus,
   FEniCS, Devito) e o solver não traz nada, **diga isso**. Prefiro descartar um
   candidato agora a descobrir no parecer.

## Direções que me parecem promissoras (mas não se limite a elas)

- **Meios heterogêneos**: condução ou transporte em material composto,
  estratificado ou com inclusões — o solver trata interfaces com forma
  conservativa de fluxo, e há resultado físico de interesse (condutividade
  efetiva, homogeneização).
- **Formação de padrões em reação-difusão** (Turing, Gray–Scott, Brusselator) —
  precisa de domínio grande para estatística decente, o que casa com a GPU.
- **Meios excitáveis** (FitzHugh–Nagumo, Aliev–Panfilov, monodomínio cardíaco) —
  propagação de frente, reentrada espiral.
- **Quimiotaxia** (Keller–Segel) e agregação.
- **Transporte em meio poroso** com heterogeneidade espacial.

## Formato da resposta

Uma seção por candidato, mais uma **recomendação final de um** com justificativa
em dois parágrafos: por que ele é o mais forte e qual o maior risco dele.

Se sua conclusão for que **nenhum** candidato justifica CPC com este envelope,
diga isso claramente e sugira o veículo adequado. Essa é uma resposta aceitável
e útil.

## Regras

- **Verifique antes de afirmar.** Se não tiver certeza de que um artigo existe
  ou de que uma equação cabe no envelope, diga que não tem certeza.
- Nada de DOI inventado. Se não achou, escreva "não localizei".
- Prefira problemas cuja física seja verificável contra algo publicado.
