from ASTROMER.models import SingleBandEncoder


model = SingleBandEncoder()
model = model.from_pretraining('macho')

import numpy as np

samples_collection = [ np.array([[5200, 0.3, 0.2],
                                 [5300, 0.5, 0.1],
                                 [5400, 0.2, 0.3]]),

                       np.array([[4200, 0.3, 0.1],
                                 [4300, 0.6, 0.3]]) ]


attention_vectors = model.encode(samples_collection,
                                 oids_list=['1', '2'],
                                 batch_size=1,
                                 concatenate=True)

print(attention_vectors)