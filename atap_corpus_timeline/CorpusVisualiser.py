import logging
import traceback
from typing import Optional

import panel
import panel as pn
from atap_corpus.corpus.corpus import DataFrameCorpus
from atap_corpus_loader import CorpusLoader
from pandas import DataFrame, Series, Grouper
from pandas.core.dtypes.common import is_datetime64_any_dtype
from panel.pane import Plotly, Markdown
from panel.widgets import Select, DatetimeRangePicker, Button, IntInput
import plotly.express as px
from plotly.graph_objs import Figure


class CorpusVisualiser(pn.viewable.Viewer):
    TIME_PERIOD_GROUPINGS: dict[str, str] = {
        'years': 'Y', 'quarters': 'Q', 'months': 'M',
        'weeks': 'W', 'days': 'D', 'hours': 'h',
        'minutes': 'min', 'seconds': 's'
    }
    TIME_PERIOD_INVERSE_GROUPINGS: dict[str, str] = {v: k for k, v in TIME_PERIOD_GROUPINGS.items()}
    CONTROLS_MAX_WIDTH: int = 200
    PLOT_LINE_WIDTH: int = 2
    PLOT_MARKER_SIZE: int = 6
    MAX_DISPLAYABLE_VALUES: int = 1000

    def log(self, msg: str, level: int):
        logger = logging.getLogger(self.logger_name)
        logger.log(level, msg)

    def __init__(self, corpus_loader: CorpusLoader, logger_name: str, **params):
        super().__init__(**params)
        self.corpus_loader: CorpusLoader = corpus_loader
        self.logger_name: str = logger_name

        self.corpus_selector = Select(name="Selected corpus", width=self.CONTROLS_MAX_WIDTH)
        self.time_col_selector = Select(name="Time metadata (x-axis)", width=self.CONTROLS_MAX_WIDTH)
        self.date_range_picker = DatetimeRangePicker(name="Selected time range", width=self.CONTROLS_MAX_WIDTH)
        self.date_group_periods = IntInput(name="Number of intervals", start=1, value=1, width=self.CONTROLS_MAX_WIDTH)
        self.date_group_unit_selector = Select(name="Intervals", options=self.TIME_PERIOD_GROUPINGS, width=self.CONTROLS_MAX_WIDTH)
        self.timeline_meta_selector = Select(name="Timeline metadata (y-axis)", width=self.CONTROLS_MAX_WIDTH)
        self.histogram_meta_selector = Select(name="Histogram metadata (x-axis)", width=self.CONTROLS_MAX_WIDTH)
        self.histogram_stack_meta_selector = Select(name="Histogram second metadata", width=self.CONTROLS_MAX_WIDTH)

        self.generate_plots_button = Button(name="Generate plots", button_style='solid', button_type='primary', width=self.CONTROLS_MAX_WIDTH)
        self.controls = pn.Column(
            self.corpus_selector,
            self.time_col_selector,
            self.date_range_picker,
            Markdown("**Frequency timeline**"),
            self.date_group_periods,
            self.date_group_unit_selector,
            self.timeline_meta_selector,
            Markdown("**Count histogram**"),
            self.histogram_meta_selector,
            self.histogram_stack_meta_selector,
            self.generate_plots_button,
            width=self.CONTROLS_MAX_WIDTH
        )

        self.frequency_plot: Plotly = Plotly(sizing_mode='stretch_width', visible=False)
        self.count_histogram: Plotly = Plotly(sizing_mode='stretch_width', visible=False)

        self.plots = pn.Column(
            self.frequency_plot,
            self.count_histogram,
            min_width=1200,
            sizing_mode='stretch_width')

        self.panel = pn.Row(
            self.controls,
            self.plots,
            min_width=1400,
            sizing_mode='stretch_width'
        )

        self.corpus_selector.param.watch(self._update_selected_corpus, ['value'])
        self.time_col_selector.param.watch(self._update_time_column, ['value'])
        self.generate_plots_button.on_click(self.generate_plots)

        self.corpus_loader.register_event_callback("build", self._update_corpus_list)
        self.corpus_loader.register_event_callback("rename", self._update_corpus_list)
        self.corpus_loader.register_event_callback("delete", self._update_corpus_list)

        panel.state.add_periodic_callback(self._update_corpus_list)

    def __panel__(self):
        return self.panel.servable()

    def display_error(self, error_msg: str):
        self.log(f"Error displayed: {error_msg}", logging.ERROR)
        panel.state.notifications.error(error_msg, duration=0)

    def display_warning(self, warning_msg: str):
        self.log(f"Warning displayed: {warning_msg}", logging.ERROR)
        panel.state.notifications.warning(warning_msg, duration=6000)

    def display_success(self, success_msg: str):
        self.log(f"Success displayed: {success_msg}", logging.INFO)
        panel.state.notifications.success(success_msg, duration=3000)

    def _update_corpus_list(self, *_):
        try:
            corpus_options: dict[str, DataFrameCorpus] = self.corpus_loader.get_corpora()
            if self.corpus_selector.options != corpus_options:
                self.corpus_selector.options = corpus_options
                if len(corpus_options):
                    self.corpus_selector.value = list(corpus_options.values())[-1]
        except Exception as e:
            self.log(str(traceback.format_exc()), logging.DEBUG)

    def _update_selected_corpus(self, *_):
        try:
            corpus: Optional[DataFrameCorpus] = self.corpus_selector.value
            if corpus is None:
                self.time_col_selector.value = None
                self.timeline_meta_selector.value = None
            else:
                meta_list: list[str] = corpus.metas
                datetime_metas: list[str] = [col for col in meta_list if is_datetime64_any_dtype(corpus.get_meta(col))]
                if self.time_col_selector.options != datetime_metas:
                    self.time_col_selector.options = datetime_metas
                    if len(datetime_metas):
                        self.time_col_selector.value = datetime_metas[0]
                selectable_dict: dict = {'': None}
                selectable_dict.update({col: col for col in meta_list if col not in datetime_metas})
                if self.timeline_meta_selector.options != selectable_dict:
                    self.timeline_meta_selector.options = selectable_dict
                    self.timeline_meta_selector.value = list(selectable_dict.values())[0]
                if self.histogram_meta_selector.options != meta_list:
                    self.histogram_meta_selector.options = meta_list
                    if len(meta_list):
                        self.histogram_meta_selector.value = meta_list[0]
                stack_meta_list: list[str] = [None] + meta_list
                if self.histogram_stack_meta_selector.options != stack_meta_list:
                    self.histogram_stack_meta_selector.options = stack_meta_list
                    if len(stack_meta_list):
                        self.histogram_stack_meta_selector.value = None
        except Exception as e:
            self.log(str(traceback.format_exc()), logging.DEBUG)

    def _update_time_column(self, *_):
        try:
            corpus: Optional[DataFrameCorpus] = self.corpus_selector.value
            time_col: Optional[str] = self.time_col_selector.value
            self.date_range_picker.value = None
            if (corpus is None) or (time_col is None) or (time_col not in corpus.metas):
                return
            date_series: Series = corpus.get_meta(time_col)
            self.date_range_picker.start = date_series.min()
            self.date_range_picker.end = date_series.max()
            self.date_range_picker.value = (date_series.min(), date_series.max())
        except Exception as e:
            self.log(str(traceback.format_exc()), logging.DEBUG)

    def generate_plots(self, *_):
        try:
            if (self.corpus_selector.value is None) or (self.time_col_selector.value is None):
                return

            corpus: DataFrameCorpus = self.corpus_selector.value
            plot_df: DataFrame = corpus.to_dataframe()
            time_col: str = self.time_col_selector.value
            timeline_meta_col: str = self.timeline_meta_selector.value
            histogram_meta_col: str = self.histogram_meta_selector.value
            histogram_stacked_meta_col: str = self.histogram_stack_meta_selector.value

            start_date, end_date = self.date_range_picker.value
            mask = (plot_df[time_col] >= start_date) & (plot_df[time_col] <= end_date)
            filtered_df = plot_df.loc[mask]

            try:
                self.frequency_plot.object = self.create_meta_frequency_plot(filtered_df, time_col, timeline_meta_col)
                self.frequency_plot.visible = True
            except ValueError as e:
                self.frequency_plot.object = None
                self.frequency_plot.visible = False
                self.display_warning(str(e))

            if histogram_meta_col is not None:
                try:
                    self.count_histogram.object = self.create_meta_count_histogram(filtered_df, histogram_meta_col, histogram_stacked_meta_col)
                    self.count_histogram.visible = True
                except ValueError as e:
                    self.count_histogram.object = None
                    self.count_histogram.visible = False
                    self.display_warning(str(e))
            else:
                self.count_histogram.object = None
                self.count_histogram.visible = False
        except Exception as e:
            self.log(str(traceback.format_exc()), logging.DEBUG)

    def create_meta_frequency_plot(self, plot_df: DataFrame, time_col: str, meta_col: Optional[str]) -> Figure:
        """
        Creates a line plot of time by frequency within the specified meta column
        """
        self.log(f"create_meta_frequency_plot method: plot_df: {plot_df.shape}, time_col: {time_col}, meta_col: {meta_col}", logging.DEBUG)

        date_grouping_periods = self.date_group_periods.value
        date_group_unit = self.date_group_unit_selector.value
        frequency: str = f"{date_grouping_periods} {date_group_unit}"
        date_group_readable = f"{date_grouping_periods} {CorpusVisualiser.TIME_PERIOD_INVERSE_GROUPINGS[date_group_unit]}"

        if meta_col is None:
            count_col_name: str = 'document_frequency'
            color = None
            groups = [Grouper(key=time_col, freq=frequency)]
            title = f"Document frequency timeline by {date_group_readable} intervals"
        else:
            num_unique_meta: int = len(plot_df[meta_col].unique())
            if num_unique_meta > CorpusVisualiser.MAX_DISPLAYABLE_VALUES:
                raise ValueError(f"Cannot display timeline because {meta_col} column contains too many distinct values")

            count_col_name: str = f'{meta_col}_frequency'
            color = meta_col
            groups = [Grouper(key=time_col, freq=frequency), meta_col]
            title = f"{meta_col} frequency timeline by {date_group_readable} intervals"

        grouped_df = plot_df.groupby(groups).size().reset_index(name=count_col_name)
        fig = px.line(grouped_df, x=time_col, y=count_col_name, color=color, title=title)
        fig.update_traces(mode="lines+markers", marker=dict(size=CorpusVisualiser.PLOT_MARKER_SIZE), line=dict(width=CorpusVisualiser.PLOT_LINE_WIDTH))

        return fig

    def create_meta_count_histogram(self, plot_df: DataFrame, meta_col: str, stacked_meta_col: Optional[str]) -> Figure:
        """
        Creates a histogram of the count of selected metadata over the dataframe
        """
        self.log(f"create_meta_count_histogram method: plot_df: {plot_df.shape}, meta_col: {meta_col}, stacked_meta_col: {stacked_meta_col}", logging.DEBUG)

        num_unique_meta: int = len(plot_df[meta_col].unique())
        if num_unique_meta > CorpusVisualiser.MAX_DISPLAYABLE_VALUES:
            raise ValueError(f"Cannot display histogram because {meta_col} column contains too many distinct values")

        group_list = [meta_col]
        colour_col = meta_col
        if stacked_meta_col:
            group_list.append(stacked_meta_col)
            colour_col = stacked_meta_col
        count_col_name: str = f'{meta_col}_count'
        grouped_df = plot_df.groupby(group_list).size().reset_index(name=count_col_name)

        fig = px.histogram(grouped_df, x=meta_col, y=count_col_name, color=colour_col, title=f"{meta_col} total count histogram")

        return fig
