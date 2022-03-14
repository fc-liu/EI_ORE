import torch

# remain_relation = {"276": 0, "R276": 1, "31": 2, "R31": 2, '17': 3, 'R17': 3,
#                    '47': 4, 'R47': 4, '161': 5, '36': 6, 'R36': 6, '57': 7, '40': 8, 'R40': 8, '463': 9}
remain_relation = {"276": 0,'131':1, "31": 2, '17': 3, 
                   '47': 4,  '161': 5, '36': 6,  '57': 7, '40': 8,  '463': 9}
nyt_rel_id = {
    "/location/location/containedby": 0,
    '/business/person/company': 1,
    '/people/person/place_lived': 2,
    '/people/person/nationality': 3,
    '/location/neighborhood/neighborhood_of': 4,
    '/book/author/works_written': 5,
    '/people/person/place_of_birth': 6,
    '/book/written_work/subjects': 7,
    '/people/deceased_person/place_of_death': 8,
    "/organization/parent/child": 9
}


def position2mask(position, max_length, byte=False, negation=False):
    """
    Position to Mask, position start from 0
    :param position:     (batch, )
        tensor([ 1,  2,  0,  3,  4])
    :param max_length:  int, max length
        5
    :param byte:        Return a ByteTensor if True, else a Float Tensor
    :param negation:
        False:
            tensor([[ 0.,  1.,  0.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0.],
                    [ 1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.]])
        True:
            tensor([[ 1.,  0.,  1.,  1.,  1.],
                    [ 1.,  1.,  0.,  1.,  1.],
                    [ 0.,  1.,  1.,  1.,  1.],
                    [ 1.,  1.,  1.,  0.,  1.],
                    [ 1.,  1.,  1.,  1.,  0.]])
    :return:
        ByteTensor/FloatTensor
    """
    batch_size = position.size(0)
    try:
        assert max_length >= torch.max(position).item()+1
    except Exception as e:
        print(e)
        return
    assert torch.min(position).item() >= 0

    range_i = torch.arange(0, max_length, dtype=torch.long).expand(
        batch_size, max_length).to(position.device)

    batch_position = position.unsqueeze(-1).expand(batch_size, max_length)

    if negation:
        mask = torch.ne(batch_position, range_i)
    else:
        mask = torch.eq(batch_position, range_i)

    if byte:
        return mask.detach()
    else:
        return mask.float().detach()
