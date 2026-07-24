import unittest

import torch
import torch.nn.functional as F

from relfx.loss import NTXentLoss


class HistoricalLossTest(unittest.TestCase):
    def test_positive_is_duplicated_in_denominator(self):
        z_i = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
        z_j = F.normalize(torch.tensor([[0.8, 0.2], [0.1, 0.9]]), dim=-1)
        temperature = 0.15

        actual = NTXentLoss(temperature)(z_i, z_j)

        z = torch.cat([z_i, z_j], dim=0)
        sim = (z @ z.T) / temperature
        positive_indices = torch.tensor([2, 3, 0, 1])
        terms = []
        for anchor, positive in enumerate(positive_indices):
            off_diagonal = torch.arange(4) != anchor
            denominator = (
                torch.exp(sim[anchor, positive])
                + torch.exp(sim[anchor, off_diagonal]).sum()
            )
            terms.append(-sim[anchor, positive] + torch.log(denominator))
        expected = torch.stack(terms).mean()

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
