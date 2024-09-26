from juxtorpus import Jux
from wordcloud import WordCloud
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
from tmtoolkit.bow.bow_stats import tfidf
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import panel.widgets as pnw
import re
from functools import partial
from juxtorpus.viz.corpus import dtm2tfidf, _wordcloud

#from atap_corpus_loader import CorpusLoader

hv.extension('bokeh')


# Visualising Jux
import panel as pn
import panel.widgets as pnw
from juxtorpus import Jux
# from juxtorpus.viz.corpus import _wordcloud
from wordcloud import WordCloud
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np

def corpus_freq_list(corpus, dtm_name='tokens', metric='tf', stopwords: list[str] = None):
    if not dtm_name in corpus.dtms.keys():
        raise ValueError(f"DTM {dtm_name} does not exist. Must be one of {', '.join(corpus.dtms.keys())}.")
    if stopwords is None: stopwords = list()
    dtm = corpus.dtms[dtm_name]
    with dtm.without_terms(stopwords) as dtm:
        if metric == 'tfidf':
            counter = dtm2tfidf(dtm)
        else:
            counter = dtm.freq_table().to_dict()
    counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
    return counter

def visualise_jux(corpora: dict, fixed_stopwords: list = []):
    if type(corpora) is not dict:
        corpora = corpora.get_corpora()

    jux_error = np.load('./app/jux_error_msg.npz')['message']
    jux_error_img = hv.RGB(jux_error).opts(
        width=jux_error.shape[1], 
        height=jux_error.shape[0], 
        xaxis=None, 
        yaxis=None
        )
    
    # Argument Set
    jux_methods = ['tf', 'tfidf', 'log_likelihood']
    dtm_types = ['tokens']

    global exclude_words
    exclude_words = fixed_stopwords
    global choice_No
    choice_No = 15
    
    # Dropdown for Corpus selection
    corpus_list = list(corpora.keys())
    corpus_A_dropdown = pnw.Select(name='Corpus_A', options=corpus_list, value=corpus_list[-2], width=210)
    corpus_B_dropdown = pnw.Select(name='Corpus_B', options=corpus_list, value=corpus_list[-1], width=210)
    
    method_dropdown = pnw.Select(name='Method', options=jux_methods, value=jux_methods[0], width=150)
    wordNo_input = pnw.IntInput(name='Word Number', value=100, step=10, start=30, end=150, width=90)
    dtm_dropdown = pnw.Select(name='Type', options=dtm_types, value=dtm_types[0], width=150)

    excl_words = set()
    excl_choice = pnw.MultiChoice(name="Words to be excluded, click to select or type to filter from the top words", options=list(excl_words), value=[], align="end", width=500, height=120)
    excl_input = pnw.TextAreaInput(placeholder='For specific word to be excluded, enter as delimitered text', width=200, height=80)

    refresh_btn = pnw.Button(name="Process", button_type="success", button_style="solid", align="end")

    wordcloud_A = pn.pane.HoloViews()
    wordcloud_B = pn.pane.HoloViews()
    wordcloud_Jux = pn.pane.HoloViews()
    jux_Legend = pn.pane.Markdown(width=400)

    @pn.depends(corpus_A_dropdown.param.value, corpus_B_dropdown.param.value, method_dropdown.param.value, dtm_dropdown.param.value, watch=True)
    def reset_choice(corpus_A_name, corpus_B_name, metric, dtm_name):
        corpus_A = corpora[corpus_A_name]
        corpus_B = corpora[corpus_B_name]
        token_no = choice_No
        freq_list_A = corpus_freq_list(corpus_A, 
                                       dtm_name=dtm_name, 
                                       metric=metric, 
                                       stopwords = fixed_stopwords)
        freq_list_B = corpus_freq_list(corpus_B, 
                                       dtm_name=dtm_name, 
                                       metric=metric, 
                                       stopwords = fixed_stopwords)
        
        options = list(freq_list_A.keys())[:token_no] + list(freq_list_B.keys())[:token_no]
        excl_choice.options = list(set(options))
        excl_choice.value = []
    
    @pn.depends(excl_input, watch=True)
    def update_choice(event):
        if excl_input.value:
            words = set([w for w in re.split(';|,|\/|\s', excl_input.value) if w])
            # Union of input words and existing choices/values
            if words.intersection(excl_choice.options):
                new_options = list(words.union(excl_choice.options))
                excl_choice.options = new_options
            if words.intersection(excl_choice.value):
                new_values = list(words.union(excl_choice.value))
                excl_choice.value = new_values
            excl_input.value = ''

    @pn.depends(excl_choice, watch=True)
    def update_stopwords(event):
        global exclude_words
        exclude_words = list(set(fixed_stopwords).union(excl_choice.value))

    # Function to update dtm dropdown options based on selected corpora
    def update_dtm_dropdown(corpus_a_name, corpus_b_name):
        curr_dtm = dtm_dropdown.value
        available_dtms = list(set(corpora[corpus_a_name].dtms.keys()).intersection(corpora[corpus_b_name].dtms.keys()))
        dtm_dropdown.options = sorted(available_dtms)
        dtm_dropdown.value = curr_dtm

    # Watch changes to both corpus changes and update dtm dropdown accordingly
    corpus_A_dropdown.param.watch(lambda event: update_dtm_dropdown(event.new, corpus_B_dropdown.value), 'value')
    corpus_B_dropdown.param.watch(lambda event: update_dtm_dropdown(corpus_A_dropdown.value, event.new), 'value')
    
    # Watch changes on the Stopwords choice and new inputs for stopwords
    excl_choice.param.watch(update_stopwords, 'value')

    # Function to generate a word cloud from a dictionary and convert it to an RGB image
    def generate_wordcloud_image(corpus, metric: str = 'tf', max_words: int = 50, dtm_name: str = 'tokens',
                stopwords: list[str] = None):
        wc = _wordcloud(corpus, metric = metric, max_words = max_words, dtm_name = dtm_name, stopwords = stopwords)
        plt.axis("off")
        return np.array(wc.to_image())

    # Function to generate word clouds based on the Corpus selection
    def display_wordcloud(corpus_name):
        corpus = corpora[corpus_name]
        selected_method = method_dropdown.value

        if selected_method not in ['tf', 'tfidf']: 
            selected_method = 'tf'
        # Generate word cloud images for selected Corpus
        corpus_wc_image = generate_wordcloud_image(corpus, 
                                                  metric=selected_method, 
                                                  max_words=wordNo_input.value, 
                                                  dtm_name=dtm_dropdown.value, 
                                                  stopwords=exclude_words)
        # Create HoloViews elements for the word clouds
        corpus_wc = hv.RGB(corpus_wc_image).opts(
            title=f'Corpus A: {corpora[corpus_name].name} -- {selected_method}', 
            width=600, height=400,
            xaxis=None, yaxis=None)
        return corpus_wc

    def display_jux_wordcloud(jux_error_img):
        corpus_a = corpus_A_dropdown.value
        corpus_b = corpus_B_dropdown.value
        selected_method = method_dropdown.value
        selected_wordno = wordNo_input.value
        dtm_name = dtm_dropdown.value
        # Run Jux among selected corpora
        if corpus_a == corpus_b:
            jux_cloud = pn.pane.Markdown(info, width=700, height=200)
        else:
            try:
                jux = Jux(corpora[corpus_a], corpora[corpus_b])
                pwc = jux.polarity.wordcloud(metric=selected_method, top=selected_wordno, dtm_names=dtm_name, stopwords=exclude_words, return_wc=True)  # change this to 'tfidf' or 'log_likelihood'
                # Create HoloViews elements for the word clouds
                pwc_array = np.array(pwc.wc.to_image())
                jux_cloud = hv.RGB(pwc_array).opts(
                    title=f'Jux between Corpus "{corpus_a}" and Corpus "{corpus_b}" -- {selected_method}', 
                    width=800, height=500,
                    xaxis=None, yaxis=None)
            except ValueError:
                jux_cloud = jux_error_img
        return jux_cloud   
        
    # Define the Jux legend text
    def jux_legend():
        corpus_a_name = corpora[corpus_A_dropdown.value].name
        corpus_b_name = corpora[corpus_B_dropdown.value].name
        method = method_dropdown.value
        
        legend_texts = {
            'tf':{'size': 'Polarised and Rare', 'solid': 'Higher frequency to one corpus', 'translucent': 'Similar frequency'},
            'tfidf': {'size': 'Tfidf of both', 'solid': 'Higher Tfidf to one corpus', 'translucent': 'Similar tfidf'},
            'log_likelihood': {'size': 'Polarised and Rare', 'solid': 'Higher log likelihood to one corpus', 'translucent': 'Similar log likelihood'}
                    }
        legend_text = f"""
        <span style='font-size:18px;'>     
        <span style='color:blue'>Blue Words</span>:  Corpus -- **{corpus_a_name}** <br>
        <span style='color:red'>Red Words</span>: Corpus -- **{corpus_b_name}** <br>
        Size: {legend_texts[method]['size']} <br>
        Solid: {legend_texts[method]['solid']} <br>
        Translucent: {legend_texts[method]['translucent']}
        </span>
        """
        return legend_text

    def refresh(event):
        refresh_btn.disabled = True
        wordcloud_A.object = display_wordcloud(corpus_A_dropdown.value)
        wordcloud_B.object = display_wordcloud(corpus_B_dropdown.value)
        wordcloud_Jux.object = display_jux_wordcloud(jux_error_img)
        jux_Legend.object = jux_legend()
        refresh_btn.disabled = False

    refresh_btn.on_click(refresh)
    refresh(True)

    # Set initial values for dtm dropdown
    update_dtm_dropdown(corpus_A_dropdown.value, corpus_B_dropdown.value)
    update_stopwords(excl_choice)
    reset_choice(corpus_A_dropdown.value, corpus_B_dropdown.value, method_dropdown.value, dtm_dropdown.value)


    # Combine everything into a dashboard
    layout = pn.Column(pn.Row(
                        pn.Column(pn.Row(corpus_A_dropdown, corpus_B_dropdown), pn.Row(dtm_dropdown, method_dropdown, wordNo_input)),
                        pn.layout.Divider(),
                        pn.Column(excl_input, refresh_btn), 
                        excl_choice),
                    pn.Row(wordcloud_A, wordcloud_B), 
                    pn.Row(wordcloud_Jux, jux_Legend)
                    )
    return layout.servable()

# Semtag needed functions

def load_usas_dict(usas_def_file):
    # get usas_tags definition
    #usas_def_file = './documents/semtags_subcategories.txt'
    usas_tags_dict = dict()
    # reading line by line
    with open(usas_def_file) as file_x:
        for line in file_x:
            usas_tags_dict[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]
    return usas_tags_dict
    
def usas_tags_def(tokens, usas_tags_dict) -> list:
    '''
    Function to interpret the definition of the USAS tag

    Args:
        token: the token containing the USAS tag to interpret
    '''
    all_usas_tags = []
    for token in tokens:
        try: 
            usas_tags = token._.pymusas_tags[0].split('/')
            if usas_tags[-1]=='':
                usas_tags = usas_tags[:-1]
        except: 
            usas_tags = 'Z99'.split('/')
        all_usas_tags.extend(usas_tags)

    tag_defs = []
    tags = []
    for usas_tag in all_usas_tags:
        tag = remove_symbols(usas_tag)
        if tag=='PUNCT':
            tag_defs.append(usas_tag)
        else:
            while tag not in usas_tags_dict.keys() and tag!='':
                tag=tag[:-1]
            try: tag_defs.append(usas_tags_dict[tag])
            except: tag_defs.append(usas_tag)
        tags.append(tag)
    return tag_defs

def remove_symbols(text: str) -> str:
    '''
    Function to remove special symbols from USAS tags

    Args:
        text: the USAS tag to check
    '''
    text = re.sub('m','',text)
    text = re.sub('f','',text)
    text = re.sub('%','',text)
    text = re.sub('@','',text)
    text = re.sub('c','',text)
    text = re.sub('n','',text)
    text = re.sub('i','',text)
    text = re.sub(r'([+])\1+', r'\1', text)
    text = re.sub(r'([-])\1+', r'\1', text)

    return text

