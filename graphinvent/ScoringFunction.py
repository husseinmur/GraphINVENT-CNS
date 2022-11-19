"""
This class is used for defining the scoring function(s) which can be used during
fine-tuning.
"""
# load general packages and functions
from collections import namedtuple
import torch
from rdkit import DataStructs
from rdkit.Chem import QED, AllChem
import numpy as np
import sklearn
from sklearn import svm
import joblib
import subprocess
import pandas as pd
import rdkit
from guacamol.common_scoring_functions import CNS_MPO_ScoringFunction
from keras.models import load_model
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

class ScoringFunction:
    """
    A class for defining the scoring function components.
    """
    def __init__(self, constants : namedtuple) -> None:
        """
        Args:
        ----
            constants (namedtuple) : Contains job parameters as well as global
                                     constants.
        """
        self.score_components = constants.score_components  # list
        self.score_type       = constants.score_type        # list
        self.qsar_models      = constants.qsar_models       # dict
        self.device           = constants.device
        self.max_n_nodes      = constants.max_n_nodes
        self.score_thresholds = constants.score_thresholds

        self.n_graphs         = None  # placeholder

        assert len(self.score_components) == len(self.score_thresholds), \
               "`score_components` and `score_thresholds` do not match."

    def compute_score(self, graphs : list, termination : torch.Tensor,
                      validity : torch.Tensor, uniqueness : torch.Tensor) -> \
                      torch.Tensor:
        """
        Computes the overall score for the input molecular graphs.

        Args:
        ----
            graphs (list)              : Contains molecular graphs to evaluate.
            termination (torch.Tensor) : Termination status of input molecular
                                         graphs.
            validity (torch.Tensor)    : Validity of input molecular graphs.
            uniqueness (torch.Tensor)  : Uniqueness of input molecular graphs.

        Returns:
        -------
            final_score (torch.Tensor) : The final scores for each input graph.
        """
        self.n_graphs          = len(graphs)
        contributions_to_score = self.get_contributions_to_score(graphs=graphs)

        if len(self.score_components) == 1:
            final_score = contributions_to_score[0]

        elif self.score_type == "continuous":
            final_score = contributions_to_score[0]
            for component in contributions_to_score[1:]:
                final_score *= component

        elif self.score_type == "binary":
            component_masks = []
            for idx, score_component in enumerate(contributions_to_score):
                component_mask = torch.where(
                    score_component > self.score_thresholds[idx],
                    torch.ones(self.n_graphs, device=self.device, dtype=torch.uint8),
                    torch.zeros(self.n_graphs, device=self.device, dtype=torch.uint8)
                )
                component_masks.append(component_mask)

            final_score = component_masks[0]
            for mask in component_masks[1:]:
                final_score *= mask
                final_score  = final_score.float()

        else:
            raise NotImplementedError

        # remove contribution of duplicate molecules to the score
        final_score *= uniqueness

        # remove contribution of invalid molecules to the score
        final_score *= validity

        # remove contribution of improperly-terminated molecules to the score
        final_score *= termination

        return final_score

    def get_contributions_to_score(self, graphs : list) -> list:
        """
        Returns the different elements of the score.

        Args:
        ----
            graphs (list) : Contains molecular graphs to evaluate.

        Returns:
        -------
            contributions_to_score (list) : Contains elements of the score due to
                                            each scoring function component.
        """
        contributions_to_score = []



        for score_component in self.score_components:
            if "target_size" in score_component:

                target_size  = int(score_component[12:])

                assert target_size <= self.max_n_nodes, \
                       "Target size > largest possible size (`max_n_nodes`)."
                assert 0 < target_size, "Target size must be greater than 0."

                target_size *= torch.ones(self.n_graphs, device=self.device)
                n_nodes      = torch.tensor([graph.n_nodes for graph in graphs],
                                            device=self.device)
                max_nodes    = self.max_n_nodes
                score        = (
                    torch.ones(self.n_graphs, device=self.device)
                    - torch.abs(n_nodes - target_size)
                    / (max_nodes - target_size)
                )
                contributions_to_score.append(score)

            elif score_component == "QED":
                mols = [graph.molecule for graph in graphs]

                # compute the QED score for each molecule (if possible)
                qed = []
                for mol in mols:
                    try:
                        qed.append(QED.qed(mol))
                    except:
                        qed.append(0.0)
                score = torch.tensor(qed, device=self.device)

                contributions_to_score.append(score)
            
            elif score_component == "CNSMPO":
                CNS = CNS_MPO_ScoringFunction()
                mpo = []
                for graph in graphs:
                    try:
                        rdkit.Chem.SanitizeMol(graph.get_molecule())
                        smiles = graph.get_smiles()
                        if (len(smiles)==0):
                            mpo.append(0.0)
                            continue
                        molscore = CNS.raw_score(smiles)
                        mpo.append(molscore)
                    except:
                        mpo.append(0.0)
                score = torch.tensor(mpo, device=self.device)
                contributions_to_score.append(score)

            elif "activity" in score_component:
                #mols = [graph.molecule for graph in graphs]

                # `score_component` has to be the key to the QSAR model in the
                # `self.qsar_models` dict
                qsar_model = self.qsar_models[score_component]
                preds1     = self.compute_activity(graphs, qsar_model)
                #preds2     = self.compute_activity_keras(graphs)

                #preds = 0.65*np.array(preds1) + 0.35*np.array(preds2)

                print(preds1)

                score = torch.tensor(preds1, device=self.device)

                contributions_to_score.append(score)

            else:
                raise NotImplementedError("The score component is not defined. "
                                          "You can define it in "
                                          "`ScoringFunction.py`.")

        return contributions_to_score

    # def compute_activity(self, mols : list,
    #                      activity_model : sklearn.svm.classes.SVC) -> list:
    #     """
    #     Note: this function may have to be tuned/replicated depending on how
    #     the activity model is saved.

    #     Args:
    #     ----
    #         mols (list) : Contains `rdkit.Mol` objects corresponding to molecular
    #                       graphs sampled.
    #         activity_model (sklearn.svm.classes.SVC) : Pre-trained QSAR model.

    #     Returns:
    #     -------
    #         activity (list) : Contains predicted activities for input molecules.
    #     """
    #     n_mols   = len(mols)
    #     activity = torch.zeros(n_mols, device=self.device)

    #     for idx, mol in enumerate(mols):
    #         try:
    #             fingerprint   = AllChem.GetMorganFingerprintAsBitVect(mol,
    #                                                                   2,
    #                                                                   nBits=2048)
    #             ecfp4         = np.zeros((2048,))
    #             DataStructs.ConvertToNumpyArray(fingerprint, ecfp4)
    #             activity[idx] = activity_model.predict_proba([ecfp4])[0][1]
    #         except:
    #             pass  # activity[idx] will remain 0.0

    #     return activity

    def compute_activity(self, molecular_graphs : list, model_path: str) -> list:
        path = "./graphinvent/models/BBB/data"
        failed_idx = []   
        f = open(path+"/smiles.smi","w")
        for idx, mol in enumerate(molecular_graphs):
            try:
                rdkit.Chem.SanitizeMol(mol.get_molecule())
                smiles = mol.get_smiles()
                if (len(smiles)==0):
                    failed_idx.append(idx)
                    continue
                f.write(smiles+"\n")
            except:
                failed_idx.append(idx)
        f.close()
        gen_features = ['java', '-jar', './graphinvent/models/BBB/PaDEL-Descriptor/PaDEL-Descriptor.jar','-descriptortypes', './graphinvent/models/BBB/PaDEL-Descriptor/descriptors.xml', '-dir', path, '-file', path+'/feature.csv', '-2d', '-fingerprints', '-removesalt', '-detectaromaticity', '-standardizenitro', '-usefilenameasmolname', '-retainorder']
        while True:
            try:
                subprocess.run(gen_features,timeout=60)
            except:
                print("PaDEL timed-out, trying again...")
                continue
            break
        features = pd.read_csv(path+"/feature.csv",error_bad_lines=False)
        features = features[['AATS4s','TopoPSA', 'GATS8s']]
        features.fillna(0,inplace=True)
        model = joblib.load(model_path)
        preds = model.predict_proba(features)[:,1]
        preds = self.insert_zeros(preds,failed_idx,len(molecular_graphs))
        return (preds)

    def insert_zeros(self,preds:list,failed:list,length:int) -> list:
        score = []
        counter = 0
        for i in range(length):
            if i in failed:
                score.append(0)
            else:
                score.append(preds[counter])
                counter +=1
        return score
    
    def compute_activity_keras(self, molecular_graphs : list) -> list:
        wmodel = word2vec.Word2Vec.load('./graphinvent/models/model_300dim.pkl')
        loaded_model = load_model("./graphinvent/models/a_sync") 
        preds = []
        for mol in molecular_graphs:
            try:
                rdkit.Chem.SanitizeMol(mol.get_molecule())
                smiles = mol.get_smiles()
                if (len(smiles)==0):
                    preds.append(0)
                vec = []
                unseen_vec = wmodel.wv.word_vec('UNK')    
                sentence = MolSentence(mol2alt_sentence(rdkit.Chem.MolFromSmiles(smiles), 1))
                keys = set(wmodel.wv.vocab.keys())
                vec.append(sum([wmodel.wv.word_vec(y) if y in set(sentence) & keys else unseen_vec for y in sentence]))
                score = loaded_model.predict(np.array(vec))[0,1]
                preds.append(score)
            except:
                preds.append(0)
        preds = np.array(preds)
        return (preds)