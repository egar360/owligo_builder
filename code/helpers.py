"""Random helper functions."""

import re
from itertools import chain
from typing import Iterable, Any
from random import shuffle, choice
import random

import numpy as np
import pandas as pd

from Bio.Seq import Seq
from Bio import SeqIO


def unique_orthogonal(sites) -> bool:
    wc_sites = [str(Seq(s).reverse_complement()) for s in sites]

    return len(set(sites.tolist() + wc_sites)) == len(sites)*2

def dna_contains_seq(dna: str, *seq: str, reverse_complement: bool = True) -> bool:
    """Checks if dna sequence contains defined sequences.
    True if sequence is present. If reverse_complement is True, then
    searches for reverse complement as well.
    """

    if reverse_complement:
        pattern = "|".join(list(chain(*[[s, str(Seq(s).reverse_complement())] for s in seq])))
    else:
        pattern = "|".join(list(seq))

    match = re.search(pattern, dna, flags=re.I)

    return match is not None

def dynamic_chunker(iterable: Iterable[Any], chunk_sizes: list[int]) -> Iterable[list[Any]]:
    """Break up list into variable-sized chunks specified in `chunk_sizes`"""

    it = iter(iterable)

    for chunk in chunk_sizes:
        yield [next(it) for i in range(chunk)]

def clean_dna(length: int, *seq: str, flank_left: str = "", flank_right: str = ""):
    """Generate random dna that avoids `seq` (and its reverse complement), including
    where it would be formed by combining with the sequence immediately upstream
    (`flank_left`) or downstream (`flank_right`) of the generated region once assembled.

    Only pass the bases of the flank actually close to the boundary (at most
    `max(len(s) for s in seq) - 1` of them) - not the whole neighboring sequence.
    A flank that itself already legitimately contains one of `seq` (e.g. the enzyme
    site the flank was built around) would otherwise make every candidate look
    contaminated and loop forever.
    """

    dna = ""
    while len(dna) < length:
        candidate = dna + random.choice(list('ATGC'))
        if not dna_contains_seq(flank_left + candidate + flank_right, *seq, reverse_complement=True):
            dna = candidate
        else:
            continue

    return dna

def random_dna(length: int):

    return "".join([choice('ATGC') for _ in range(length)])

def flatten(iterable: list) -> list:
    """Remove one level of a nested list."""

    return list(chain(*iterable))

def index_array(iterable: Iterable[int], length: int) -> np.ndarray:
    """Return array with 1's at the indexes included in iterable."""

    return np.array([1 if i in iterable else 0 for i in range(length)])

def read_fasta(file: str) -> pd.DataFrame:
    """Returns dataframe of sequences from fasta file. Preserves
    all information included in fasta file.
    
    Args:
        file (str): relative file path

    Returns:
        pd.DataFrame with `sequence` column holding sequences. Index is automatically
        0 to n-sequences, meaning that this format does not accept custom labels.
    """

    records = list(SeqIO.parse(file, format='fasta'))
    records = [
        pd.DataFrame({
            'sequence':[str(r.seq)], 'fasta_id':[r.id], 'fasta_description':[r.description]
        }) for r in records
    ]

    return pd.concat(records, ignore_index=True)

def overlap_split(iterable, overlap=1) -> list:
    """Returns a nested list of split input iterable with n elements
    overlapping between each sublist. No unique elements except
    first and last element of input iterable.
    
    Args:
        iterable: the iterable to split
        overlap: number of elements to repeat between each sublist
    
    Returns:
        nested list
    """

    return [[iterable[i:i+1+overlap]] for i in range(len(iterable)-overlap)]

def shuffler(iterable):
    iterable = iterable.copy()
    shuffle(iterable)

    return iterable
