## Installation

```pip install uv```

```uv pip install -r requirements2.txt```


```pip install flash-attn==2.6.3 --no-build-isolation```

## Prepare for Lean and Mathlib
You can dowload lean via

```curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh```

```export PATH="$HOME/.elan/bin:$PATH"```

```elan toolchain install leanprover/lean4:4.9.0-rc1```

Set 4.9.0-rc1 as the default for this environment

```elan default leanprover/lean4:4.9.0-rc1```

```git clone https://github.com/xinhjBrant/mathlib4.git```

Install mathlib4

```cd mathlib4 ```

```git fetch origin 2f65ba7f1a9144b20c8e7358513548e317d26de1```

```git checkout 2f65ba7f1a9144b20c8e7358513548e317d26de1```

```cd mathlib4```

Dependency Configuration (lakefile.lean)
Ensure your lakefile.lean includes the following requirements with the exact versions/commits. This is necessary for the REPL tactic and other utilities to function correctly.

```require batteries from git "https://github.com/leanprover-community/batteries" @ "42b5dddbd6b2658fcfede9dad26cc47737edec2d"
require Qq from git "https://github.com/leanprover-community/quote4" @ "a7bfa63f5dddbcab2d4e0569c4cac74b2585e2c6"
require aesop from git "https://github.com/leanprover-community/aesop" @ "7e3bd939c6badfcb1e607c0fddb509548baafd05"
require proofwidgets from git "https://github.com/leanprover-community/ProofWidgets4" @ "v0.0.36"
require Cli from git "https://github.com/leanprover/lean4-cli" @ "2cf1030dc2ae6b3632c84a09350b675ef3e347d0"
require importGraph from git "https://github.com/leanprover-community/import-graph.git" @ "7983e959f8f4a79313215720de3ef1eca2d6d474"
require REPL from git "https://github.com/xinhjBrant/repl.git" @ "master"
```

Then you should set mathlib4 by using following commands

```lake update```

```lake build```

```lake update REPL```

## Library Dependencies
If your trl version is old, you might have to set upper bound of clip as
```
epsilon_high=self.epsilon+0.08
coef_1 = torch.exp(per_token_logps - old_per_token_logps)
coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + epsilon_high)   
is_clipped = ((coef_1 < 1 - self.epsilon) & (advantages.unsqueeze(1) < 0)) | (
         (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0))
```
## Train and Evaluation
You can use ```train.sh``` or ```traub_peft.sh``` for training.
For evaluation, generate samples through ```inference.sh``` or ```peft_inference.sh``` and ```inference_step.sh``` verify them 
