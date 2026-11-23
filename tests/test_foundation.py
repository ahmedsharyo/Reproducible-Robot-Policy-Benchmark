import numpy as np
import pytest
from robot_benchmark.datasets import generate_dataset, load_dataset, save_dataset, split_episodes
from robot_benchmark.environments import ReachingEnv
from robot_benchmark.policies import BCModel
import torch

def test_deterministic_environment():
    a=ReachingEnv(); oa,_=a.reset(123); b=ReachingEnv(); ob,_=b.reset(123); np.testing.assert_array_equal(oa,ob)
    assert a.step(np.zeros(2))[0].shape==(4,)

def test_dataset_checksum_roundtrip(tmp_path):
    ds=generate_dataset(8,7); p=tmp_path/"demos.npz"; digest=save_dataset(ds,p); loaded=load_dataset(p)
    assert loaded.metadata["schema_version"]==1 and len(loaded.episodes)==8 and len(digest)==64
    train,val,test=split_episodes(loaded,3,.6,.2); assert not (set(train)&set(val)|set(train)&set(test)|set(val)&set(test))
    p.with_suffix(".npz.sha256").write_text("bad")
    with pytest.raises(ValueError): load_dataset(p)

def test_model_shapes():
    model=BCModel(hidden_sizes=(16,)); assert model(torch.zeros(3,4)).shape==(3,2)
