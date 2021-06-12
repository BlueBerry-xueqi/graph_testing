## Requirements

- pytorch
- pytorch-geometric
- deeprobust

## Dataset

- The citation network datasets "Cora", "CiteSeer" and "PubMed" from the `"Revisiting Semi-Supervised Learning with Graph Embeddings" <https://arxiv.org/abs/160308861>`
- MoleculeNet <http://moleculenet.ai/datasets-1
- PPI The protein-protein interaction networks from the `"Predicting Multicellular Function through Multi-layer Tissue Networks"  <https://arxiv.org/abs/1707.04638>`
- The Reddit dataset from the `"Inductive Representation Learning on  Large Graphs" <https://arxiv.org/abs/1706.02216>`
- A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY", "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University  <https://chrsmrrs.github.io/datasets>`_. In addition, this dataset wrapper provides `cleaned dataset versions <https://github.com/nd7141/graph_datasets>`_ as motivated by the "Understanding Isomorphism Bias in Graph Data Sets"  <https://arxiv.org/abs/1910.12091>
- Node Property Prediction, https://ogb.stanford.edu/docs/nodeprop/
- AMiner The heterogeneous AMiner dataset from the `"metapath2vec: Scalable Representation Learning for Heterogeneous Networks" <https://ericdongyx.github.io/papers/ KDD17-dong-chawla-swami-metapath2vec.pdf>`
- The Flickr dataset from the `"GraphSAINT: Graph Sampling Based Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,  containing descriptions and common properties of image
- The ZINC dataset from the `ZINC database <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the `"Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules" <https://arxiv.org/abs/1610.02415>`_ paper, containing about  250,000 molecular graphs with up to 38 heavy atoms.   The task is to regress a synthetic computed property dubbed as the constrained solubility.
- The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep  Representation for Volumetric Shapes"  <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,  containing CAD models of 10 and 40 categories, respectively.
- The ShapeNet part level segmentation dataset from the `"A ScalableActive Framework for Region Annotation in 3D Shape Collections  <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_ paper, containing about 17,000 3D shape point clouds from 16 shapecategories. Each category is annotated with 2 to 6 parts.
- The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular  Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of about 130,000 molecules with 19 regression targets. Each molecule includes complete spatial information for the single low energy conformation of the atoms in the molecule. In addition, we provide the atom features from the `"Neural Message  Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>
- The Integrated Crisis Early Warning System (ICEWS) dataset used in the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of events collected from 1/1/2018 to 10/31/2018 (24 hours time granularity).
- The relational entities networks "AIFB", "MUTAG", "BGS" and "AM" from  the `"Modeling Relational Data with Graph Convolutional Networks"  <https://arxiv.org/abs/1703.06103>`_ paper.
- The Bitcoin-OTC dataset from the `"EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs" <https://arxiv.org/abs/1902.10191>`_ paper, consisting of 138 who-trusts-whom networks of sequential time steps.
- JODIEDataset, reddit', 'wikipedia', 'mooc', 'lastfm', http://snap.stanford.edu/jodie/#datasets

## Cosider Tasking
- Graph Classification
- Node Classification
- Link Prediction

## To implement Test Metrics

- LSA/DSA
- NC
- DeepGini
- MCP
- CES