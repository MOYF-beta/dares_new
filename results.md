> abs_rel , sq_rel , rmse

|model|SyntheticColon|C3VD|SCARED|Hamlyn|comment|
|-----|--------------|----|------|------|-------|
|simplified|-|-|0.0506  , 0.3764  , 4.6368|-|only scared|
|simplified_peft_dora|-|-|0.0501  , 0.3573  , 4.5112|-|only scared|
|vit pose|-|-| 0.0546  , 0.4258  , 4.9334 |-|only scared|
|vivit pose|-|-|0.0961  , 5.7129  , 15.2935|-|something wrong|
|simplified|-|0.0545  , 0.3553  , 4.6495|-|-|only c3vd|
|simplified|Lost|-|-|-|only simcol|
|simplified|0.0969  , 0.5846  , 4.1872|0.0661  , 0.6222  , 6.0163|0.0685  , 0.6075  , 5.8801|-|no Hamlyn|
|simplified|0.1095, 1.6787, 5.4549|0.2879,7.9341, 23.0057|0.0640, 0.6101, 5.8857|0.1939,11.9786 , 23.4538|with Hamlyn|

```python
# when testing ENDO DAC like models, I didn't notice this before
out = self.motion_scale*out.view(-1, self.num_frames_to_predict_for, 1, 6)
```