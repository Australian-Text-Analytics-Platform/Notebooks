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

#from atap_corpus_loader import CorpusLoader

hv.extension('bokeh')

def dtm2tfidf(dtm):
    tfidf_mat = tfidf(dtm.matrix)
    freq = dict(zip(dtm.vocab(), sum(tfidf_mat.toarray())))
    return freq
    
def _wordcloud(corpus, max_words: int, metric: str, dtm_name: str, stopwords: list[str] = None):
    if stopwords is None: stopwords = list()
    stopwords.extend(ENGLISH_STOP_WORDS)
    dtm_names = {'tokens', 'semtag', 'hashtag'}
    metrics = {'tf', 'tfidf'}
    assert dtm_name in dtm_names, f"{dtm_name} not in {', '.join(dtm_names)}"
    assert metric in metrics, f"{metric} not in {', '.join(metrics)}"
    wc = WordCloud(background_color='white', max_words=max_words, height=600, width=1200, stopwords=stopwords)

    dtm = corpus.dtms[dtm_name]
    if dtm_name in corpus.dtms.keys():
        dtm = corpus.dtms[dtm_name]  # corpus dtm is always lower cased.
    with dtm.without_terms(stopwords) as dtm:
        if metric == 'tf':
            counter = dtm.freq_table().to_dict()
            wc.generate_from_frequencies(counter)
            return wc
        elif metric == 'tfidf':
            counter = dtm2tfidf(dtm)
            wc.generate_from_frequencies(counter)
            return wc
        else:
            raise ValueError(f"Metric {metric} is not supported. Must be one of {', '.join(metrics)}")

def visualise_versions(corpora, corpus_name):
    corpus = process_corpus(corpora, corpus_name)
    shorten_char = 500

    # Load the data
    df = corpus.to_dataframe()
    # Compute word sizes for newwords and rmwords
    df['add_token'] = df['add'].apply(lambda t: len(t.split(' ')))
    df['rm_token'] = df['rm'].apply(lambda t: -len(t.split(' ')))
    df['short_add'] = df['add'].str[:shorten_char] + np.where(df['add'].str.len() > shorten_char, '...', '')
    df['short_rm'] = df['rm'].str[:shorten_char] + np.where(df['rm'].str.len() > shorten_char, '...', '')
    # Convert the timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by="timestamp")

    x_range = (df['timestamp'].min(), df['timestamp'].max())

    # Dropdown for category selection
    category_dropdown = pnw.MultiSelect(name='Category', options=df['Category'].unique().tolist(), value=df['Category'].unique().tolist())

    # Define unique_sites and color_map
    unique_sites = sorted(df['site_hostname'].unique())
    color_map = {site: Category20c[20][i % 20] for i, site in enumerate(unique_sites)}

    # Compute the number of unique years in the dataset
    num_unique_years = len(df['timestamp'].dt.year.unique())


    # MultiSelect dropdown for site_hostname selection
    site_hostname_dropdown = pnw.MultiSelect(name='Websites', options=list(unique_sites), value=list(unique_sites), size=4)

    # Extract unique years from df
    unique_years = sorted(df['year'].unique())

    # Create a dropdown for year selection
    year_dropdown = pnw.MultiSelect(name='Year', options=unique_years, value=unique_years, size=4)  # default to all years


    # Function to update site_hostname dropdown options based on selected category
    def update_site_dropdown(category):
        relevant_sites = df[df['Category'].isin(category)]['site_hostname'].unique().tolist()
        site_hostname_dropdown.options = sorted(relevant_sites)
        site_hostname_dropdown.value = relevant_sites

    # Set initial values for site_hostname dropdown
    update_site_dropdown(category_dropdown.value)

    # Watch changes to category_dropdown and update site_hostname dropdown accordingly
    category_dropdown.param.watch(lambda event: update_site_dropdown(event.new), 'value')

    # Function to update year dropdown options based on selected category and sites
    def update_year_dropdown(category, sites):
        relevant_years = df[(df['Category'].isin(category)) & (df['site_hostname'].isin(sites))]['year'].unique().tolist()
        year_dropdown.options = sorted(relevant_years)
        year_dropdown.value = relevant_years

    # Watch changes to site_hostname_dropdown and update year_dropdown accordingly
    site_hostname_dropdown.param.watch(lambda event: update_year_dropdown(category_dropdown.value, event.new), 'value')

    # Set the initial values for year_dropdown
    update_year_dropdown(category_dropdown.value, site_hostname_dropdown.value)

    # Refactor the function to apply filtering on the original DataFrame directly and customize the tooltip
    @pn.depends(category_dropdown.param.value, site_hostname_dropdown.param.value, year_dropdown.param.value)
    def update_plot_with_custom_tooltip_refactored(selected_category, selected_sites, selected_year):
        # Filter data based on selected category and site_hostnames from the original DataFrame

        filtered_df = df[(df['Category'].isin(selected_category)) \
                           & (df['site_hostname'].isin(selected_sites)) \
                           & (df['year'].isin(selected_year))]

        # Define tooltips
        tooltips_newwords = [
            ("New Contents", "@short_add"),
            ("Time", "@timestamp{%Y-%m-%d}"),  # Adjusted format for timestamp
            ("Site", "@site_hostname")        
            ]

        tooltips_rmwords = [
            ("Removed Contents", "@short_rm"),
            ("Time", "@timestamp{%Y-%m-%d}"),  # Adjusted format for timestamp
            ("Site", "@site_hostname")
            ]

        # Bars for newwords with color mapping and custom tooltip
        bars_newwords = hv.Bars(
            filtered_df, kdims=['timestamp'], vdims=['add_token', 'short_add', 'site_hostname']
        ).opts(
            color=dim('site_hostname').categorize(color_map),
        tools=[HoverTool(tooltips=tooltips_newwords, formatters={'@timestamp': 'datetime'})]  # Explicitly format timestamp as datetime
        ).opts(
            xaxis='bottom',
            xformatter=FuncTickFormatter(code="return new Date(tick).getFullYear() + '-' + (new Date(tick).getMonth()+1).toString().padStart(2, '0');"),
            xrotation=45
            #xticks=YearsTicker(),
            #xlim=x_range
        )

        bars_rmwords = hv.Bars(
            filtered_df, kdims=['timestamp'], vdims=['rm_token', 'short_rm', 'site_hostname']
        ).opts(
            color=dim('site_hostname').categorize(color_map),
        tools=[HoverTool(tooltips=tooltips_rmwords, formatters={'@timestamp': 'datetime'})]  # Explicitly format timestamp as datetime
        ).opts(
            xaxis='bottom',
            xformatter=FuncTickFormatter(code="return new Date(tick).getFullYear() + '-' + (new Date(tick).getMonth()+1).toString().padStart(2, '0');"),
            xrotation=45
            #xticks=YearsTicker(),
            #xlim=x_range
        )


        # Overlay the two bar plots
        overlay_bars = (bars_newwords * bars_rmwords).opts(
            width=800, height=400, xlabel='Year', ylabel='Word Count Changes', show_legend=True
        ).opts(xlabel='Year')

        return overlay_bars

    # Function to generate a word cloud from a dictionary and convert it to an RGB image
    def generate_wordcloud_image(data_dict):
        if len(data_dict) > 1:
            wc = WordCloud(background_color='white').generate_from_frequencies(data_dict)
        else:
            wc = WordCloud(background_color='white').generate_from_frequencies({' ': 1})
        return np.array(wc.to_image())

    # Function to generate a combined word cloud from a DataFrame
    def generate_combined_wordcloud_image(df, column_name):
        combined_dict = Counter({})
        for index, row in df.iterrows():
            combined_dict.update(row[column_name])
        return generate_wordcloud_image(combined_dict), len(combined_dict.keys())

    # Function to generate word clouds based on the current filtered DataFrame
    @pn.depends(category_dropdown.param.value, site_hostname_dropdown.param.value, year_dropdown.param.value)
    def display_wordclouds(selected_category, selected_sites, selected_years):
        filtered_df = df[(df['Category'].isin(selected_category)) & (df['site_hostname'].isin(selected_sites)) & (df['year'].isin(selected_years))]

        # Generate word cloud images for newwords and rmwords
        newwords_image, newwords_no = generate_combined_wordcloud_image(filtered_df, 'newwords')
        rmwords_image, rmwords_no = generate_combined_wordcloud_image(filtered_df, 'rmwords')

        # Create HoloViews elements for the word clouds
        newwords_cloud = hv.RGB(newwords_image).opts(title=f'New Words: {newwords_no}', width=500, height=300)
        rmwords_cloud = hv.RGB(rmwords_image).opts(title=f'Removed Words: {rmwords_no}', width=500, height=300)

        return (newwords_cloud + rmwords_cloud).cols(1)

    # Combine everything into a dashboard
    layout = pn.Row(
        pn.Column(pn.Row(category_dropdown, site_hostname_dropdown, year_dropdown), update_plot_with_custom_tooltip_refactored),
        display_wordclouds
    )

    layout.servable()

    # Function to generate a word cloud from a dictionary and convert it to an RGB image
    def generate_wordcloud_image(data_dict):
        if len(data_dict) > 1:
            wc = WordCloud(background_color='white').generate_from_frequencies(data_dict)
        else:
            wc = WordCloud(background_color='white').generate_from_frequencies({' ': 1})
        return np.array(wc.to_image())

    # Function to generate a combined word cloud from a DataFrame
    def generate_combined_wordcloud_image(df, column_name):
        combined_dict = Counter({})
        for index, row in df.iterrows():
            combined_dict.update(row[column_name])
        return generate_wordcloud_image(combined_dict), len(combined_dict.keys())

    # Function to generate word clouds based on the current filtered DataFrame
    @pn.depends(category_dropdown.param.value, site_hostname_dropdown.param.value, year_dropdown.param.value)
    def display_wordclouds(selected_category, selected_sites, selected_years):
        filtered_df = df[(df['Category'].isin(selected_category)) & (df['site_hostname'].isin(selected_sites)) & (
            df['year'].isin(selected_years))]

        # Generate word cloud images for newwords and rmwords
        newwords_image, newwords_no = generate_combined_wordcloud_image(filtered_df, 'newwords')
        rmwords_image, rmwords_no = generate_combined_wordcloud_image(filtered_df, 'rmwords')

        # Create HoloViews elements for the word clouds
        newwords_cloud = hv.RGB(newwords_image).opts(title=f'New Words: {newwords_no}', width=500, height=300)
        rmwords_cloud = hv.RGB(rmwords_image).opts(title=f'Removed Words: {rmwords_no}', width=500, height=300)

        return (newwords_cloud + rmwords_cloud).cols(1)

    # Combine everything into a dashboard
    layout = pn.Row(
        pn.Column(pn.Row(category_dropdown, site_hostname_dropdown, year_dropdown),
                  update_plot_with_custom_tooltip_refactored),
        display_wordclouds
    )

    return layout.servable()


def visualise_jux(corpora: dict):# | CorpusLoader):
    # Visualising Jux
    import panel as pn
    import panel.widgets as pnw
    from juxtorpus import Jux
    from wordcloud import WordCloud
    import holoviews as hv
    import matplotlib.pyplot as plt
    import numpy as np

    if type(corpora) is not dict:
        corpora = corpora.get_corpora()
    
    # Dropdown for Corpus selection
    corpus_list = list(corpora.keys())
    corpus_A_dropdown = pnw.Select(name='Corpus_A', options=corpus_list, value=corpus_list[0])
    corpus_B_dropdown = pnw.Select(name='Corpus_B', options=corpus_list, value=corpus_list[1])

    # Argument Set
    jux_methods = ['tf', 'tfidf', 'log_likelihood']
    method_dropdown = pnw.Select(name='Method', options=jux_methods, value=jux_methods[0], width=150)
    wordNo_input = pnw.IntInput(name='Word Number', value=100, step=10, start=30, end=150, width=90)
    dtm_types = ['tokens']
    dtm_dropdown = pnw.Select(name='Type', options=dtm_types, value=dtm_types[0], width=150)

    # Function to update dtm dropdown options based on selected corpora
    def update_dtm_dropdown(corpus_a_name, corpus_b_name):
        curr_dtm = dtm_dropdown.value
        available_dtms = list(set(corpora[corpus_a_name].dtms.keys()).intersection(corpora[corpus_b_name].dtms.keys()))
        dtm_dropdown.options = sorted(available_dtms)
        dtm_dropdown.value = curr_dtm

    # Set initial values for dtm dropdown
    update_dtm_dropdown(corpus_A_dropdown.value, corpus_B_dropdown.value)

    # Watch changes to both corpus changes and update dtm dropdown accordingly
    corpus_A_dropdown.param.watch(lambda event: update_dtm_dropdown(event.new, corpus_B_dropdown.value), 'value')
    corpus_B_dropdown.param.watch(lambda event: update_dtm_dropdown(corpus_A_dropdown.value, event.new), 'value')
        
    # Function to generate a word cloud from a dictionary and convert it to an RGB image
    def generate_wordcloud_image(corpus, metric: str = 'tf', max_words: int = 50, dtm_name: str = 'tokens',
                stopwords: list[str] = None):
        wc = _wordcloud(corpus, metric = metric, max_words = max_words, dtm_name = dtm_name, stopwords = stopwords)
        return np.array(wc.to_image())


    # Function to generate word clouds based on the Corpus A selection
    @pn.depends(corpus_A_dropdown.param.value, method_dropdown.param.value, wordNo_input.param.value, dtm_dropdown.param.value)
    def display_wordcloud_A(selected_corpus, selected_method, selected_wordno, dtm_name):
        if selected_method not in ['tf', 'tfidf']: 
            selected_method = 'tf'
        # Generate word cloud images for both Corpus A
        corpus_A_image = generate_wordcloud_image(corpora[selected_corpus], metric=selected_method, max_words=selected_wordno, dtm_name=dtm_name)
        # Create HoloViews elements for the word clouds
        corpus_A_cloud = hv.RGB(corpus_A_image).opts(title=f'Corpus A: {corpora[selected_corpus].name} -- {selected_method}', width=600, height=400)
        return corpus_A_cloud

    @pn.depends(corpus_B_dropdown.param.value, method_dropdown.param.value, wordNo_input.param.value, dtm_dropdown.param.value)
    def display_wordcloud_B(selected_corpus, selected_method, selected_wordno, dtm_name):
        if selected_method not in ['tf', 'tfidf']:
            selected_method = 'tf'
        # Generate word cloud images for both Corpus  B
        corpus_B_image = generate_wordcloud_image(corpora[selected_corpus], metric=selected_method, max_words=selected_wordno, dtm_name=dtm_name)
        # Create HoloViews elements for the word clouds
        corpus_B_cloud = hv.RGB(corpus_B_image).opts(title=f'Corpus B: {corpora[selected_corpus].name} -- {selected_method}', width=600, height=400)
        return corpus_B_cloud

    @pn.depends(corpus_A_dropdown.param.value, corpus_B_dropdown.param.value, method_dropdown.param.value, wordNo_input.param.value, dtm_dropdown.param.value)
    def display_jux_wordcloud(corpus_a, corpus_b, selected_method, selected_wordno, dtm_name):
        # Run Jux among selected corpora
        info = f"""
            <center>
            ### Jux can not compare corpora that are identical or very balanced in distribution.
            </center>
            """
        if corpus_a == corpus_b:
            jux_cloud = pn.pane.Markdown(info, width=700, height=200)
        else:
            try:
                jux = Jux(corpora[corpus_a], corpora[corpus_b])
                pwc = jux.polarity.wordcloud(metric=selected_method, top=selected_wordno, dtm_names=dtm_name, return_wc=True)  # change this to 'tfidf' or 'log_likelihood'
                # Create HoloViews elements for the word clouds
                pwc_array = np.array(pwc.wc.to_image())
                jux_cloud = hv.RGB(pwc_array).opts(title=f'Jux between Corpus "{corpus_a}" and Corpus "{corpus_b}" -- {selected_method}', width=800, height=500)
            except ValueError:
                jux_cloud = pn.pane.Markdown(info, width=700, height=200)
        return jux_cloud   
        
    # Define the Jux legend text
    @pn.depends(corpus_A_dropdown.param.value, corpus_B_dropdown.param.value, method_dropdown.param.value)
    def jux_legend(corpus_a_name, corpus_b_name, method):
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
        # Create a Markdown pane for the legend
        legend = pn.pane.Markdown(legend_text, width=400)
        return legend

    # Combine everything into a dashboard
    layout = pn.Column(pn.Row(corpus_A_dropdown, corpus_B_dropdown, wordNo_input, method_dropdown, dtm_dropdown), 
                    pn.Row(display_wordcloud_A, display_wordcloud_B), 
                    pn.Row(display_jux_wordcloud, jux_legend))

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

