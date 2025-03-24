# Dictionary for OCTAR model

tag_vocabulary={
    0: 'opt0',
    1: 'opt1',
    2: 'opt2',
    3: 'opt3',
    4: 'exp1',
    5: 'exp2',
    6: 'rnd',
    7: 'unk'
}

tag_mapping_id={v:k for k, v in tag_vocabulary.items()}

tag_mapping_gamma={v:0.0 for v in tag_vocabulary.values()}
tag_mapping_gamma.update(
    {'opt0': 0.0,
     'opt1': 0.5,
     'opt2': 0.93,
     'opt3': 0.994
    })
