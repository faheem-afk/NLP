from utils import build_v, cleansed, labelled_pos, labeller, load_data

class DataPreprocessing():
    
    def __init__(self):
        self.df_train, self.df_test = load_data()

    def pre_processing(self):

        # cleaning the training data
        self.df_train['tokens'], self.df_train['ner_tags'], self.df_train['pos_tags'] = cleansed(build_v(self.df_train))
        # cleaning the testing data
        self.df_test['tokens'], self.df_test['ner_tags'], self.df_test['pos_tags'] = cleansed(build_v(self.df_test))
        
        pos_labelled = labelled_pos(self.df_train)
        
        # converting the pos tags into their numeric values
        self.df_train['pos_tags'] = self.df_train['pos_tags'].map(lambda x: labeller(x, pos_labelled))
        self.df_test['pos_tags'] = self.df_test['pos_tags'].map(lambda x: labeller(x, pos_labelled))


        return self.df_train, self.df_test


        













