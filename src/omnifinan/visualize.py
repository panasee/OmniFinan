from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib as ta
from langchain_core.runnables.graph import MermaidDrawMethod
from plotly.subplots import make_subplots
from pyomnix.data_process import DataManipulator
from pyomnix.omnix_logger import get_logger
from pyomnix.utils import ObjectArray

from omnifinan.analysis.indicators import cross_over, cross_under
from omnifinan.utils import compute_rangebreaks, filter_trading_days

logger = get_logger("visualize")

DEFAULT_CANDLE_SPEC = {
    "increasing": {
        "line": {
            "color": "#d42e5b",
            "width": 1,
        },
        "fillcolor": "rgba(212, 46, 91, 0.2)",
    },
    "decreasing": {
        "line": {
            "color": "#009b75",
            "width": 1,
        },
        "fillcolor": "rgba(0, 155, 117, 0.2)",
    },
}

DEFAULT_BAR_SPEC = {
    "width": 1000 * 60 * 60 * 24 * 0.8,  # default using milliseconds
    "opacity": 0.3,
}


DEFAULT_SCATTER_SPEC = {
    "mode": "lines",
    "line": {
        "color": "#2962FF",  # TradingView主指标线常用蓝色
        "width": 2,  # 线宽适中
        "dash": "solid",  # 实线
        "shape": "linear",  # 平滑曲线
    },
    "opacity": 1.0,
    "showlegend": True,
    "connectgaps": True,
    "marker": {
        "size": 0,  # 不显示点
    },
}

VOLUME_DEF = "volume"  # default to成交量


def macro_structured_to_dataframe(
    structured: dict[str, Any],
    *,
    dimension: str | None = None,
    country: str | None = None,
) -> pd.DataFrame:
    """Convert structured macro payload to plotting-friendly long DataFrame."""
    long_rows = structured.get("chart_data", {}).get("long", []) if isinstance(structured, dict) else []
    if not isinstance(long_rows, list):
        return pd.DataFrame(columns=["key", "date", "value", "dimension", "country", "source"])
    df = pd.DataFrame(long_rows)
    if df.empty:
        return pd.DataFrame(columns=["key", "date", "value", "dimension", "country", "source"])
    if dimension:
        df = df[df["dimension"] == dimension]
    if country:
        df = df[df["country"] == country]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values(["dimension", "key", "date"])
    return df


def create_macro_figure(
    structured: dict[str, Any],
    *,
    dimensions: list[str] | None = None,
    max_series_per_dimension: int = 6,
    height: int = 1100,
    width: int = 1300,
) -> go.Figure:
    """Create multi-panel macro chart from structured macro payload."""
    dims = dimensions or ["growth", "inflation", "liquidity", "credit", "market_feedback"]
    cards = structured.get("metrics", {}) if isinstance(structured, dict) else {}
    if not isinstance(cards, dict):
        cards = {}
    df = macro_structured_to_dataframe(structured)

    fig = make_subplots(
        rows=max(1, len(dims)),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=[d.title() for d in dims],
    )
    if df.empty:
        fig.update_layout(
            title="Macro Dashboard (no data)",
            template="plotly_white",
            width=width,
            height=height,
        )
        return fig

    row = 1
    for dim in dims:
        dim_df = df[df["dimension"] == dim]
        if dim_df.empty:
            row += 1
            continue
        metric_keys = [
            k
            for k, card in cards.items()
            if isinstance(card, dict) and card.get("dimension") == dim and card.get("error") is None
        ]
        if not metric_keys:
            metric_keys = list(dim_df["key"].drop_duplicates())

        metric_keys = sorted(
            metric_keys,
            key=lambda k: int(dim_df[dim_df["key"] == k].shape[0]),
            reverse=True,
        )[: max(1, max_series_per_dimension)]

        for key in metric_keys:
            sub = dim_df[dim_df["key"] == key]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["date"],
                    y=sub["value"],
                    mode="lines",
                    name=key,
                    line={"width": 2},
                ),
                row=row,
                col=1,
            )
        row += 1

    snap = structured.get("meta", {}).get("snapshot_at") if isinstance(structured, dict) else None
    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        title=f"Macro Dashboard ({snap})" if snap else "Macro Dashboard",
        legend={"orientation": "h", "y": -0.06},
        margin={"l": 40, "r": 20, "t": 60, "b": 70},
    )
    return fig


def save_graph_as_png(app, output_file_path: Path) -> None:
    png_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open(output_file_path, "wb") as f:
        f.write(png_image)


class StockFigure:
    """
    used for plotting stock data(or any similar signals),
    there are two set of data structures:
    1. n_rows * n_cols structure, each element is a subgroup of
    figures, containing a main figure and n_subfig subfigures below it
    2. [(1+n_subfig)*n_rows] * n_cols structure, the actual figure array, used only for add_trace
    """

    def __init__(
        self,
        n_rows: int = 1,
        n_cols: int = 1,
        no_subfig: int = 1,
        subfig_height: float = 0.2,
        width: int = 600,
        height: int = 1600,
        lookback_years: int | None = 5,
    ):
        self.actual_rows = (1 + no_subfig) * n_rows
        self.sub_rows = 1 + no_subfig  # rows of per group
        # the datas are stored w.r.t. subgroups not separate figures
        self.plotobj = DataManipulator(n_rows, n_cols, data_fill_value={"candle": [], "vol": []})
        self.markets = np.empty((n_rows, n_cols), dtype=object)
        self.plot_datas = self.plotobj.datas  # just an ObjectArray used for real-time plotting
        self.data_dfs = ObjectArray(n_rows, n_cols)  # used for data calculation
        self.lookback_years = lookback_years
        self.fig = self._create_fig_and_arrange_data(
            n_rows, n_cols, no_subfig, subfig_height, width, height
        )

    def _update_xrangebreaks(self, row: int, col: int) -> None:
        if self.markets[row, col] is None:
            logger.warning(
                "market is not set for row %d and col %d, so cannot update xrangebreaks", row, col
            )
            return
        if self.data_dfs[row, col] is None:
            logger.warning(
                "data_df is not set for row %d and col %d, so cannot update xrangebreaks", row, col
            )
            return
        rbs = compute_rangebreaks(
            self.data_dfs[row, col].index.min(),
            self.data_dfs[row, col].index.max(),
            self.markets[row, col],
        )
        self.fig.update_xaxes(rangebreaks=rbs, row=row + 1, col=col + 1)

    def _create_fig_and_arrange_data(
        self,
        n_rows: int = 1,
        n_cols: int = 1,
        no_subfig: int = 1,
        subfig_height: float = 0.2,
        width: int = 800,
        height: int = 1800,
    ) -> go.Figure:
        """
        Create a plotly figure with the given dimensions.
        For each subgroup, it will contain a main figure(candlestick trace and volume trace)
        and `no_subfig`(at least 1 for MACD) subfigures below
        (subfigure height is `subfig_height` of the main figure).
        Args:
            n_rows: int, number of rows of the figure.
            n_cols: int, number of columns of the figure.
            width: int, width of the figure.
            height: int, height of the figure.
            plot_types: list, list of plot types to be used.
        """
        logger.validate(no_subfig >= 1, "no_subfig must be at least 1")

        # assign heights to each row
        actual_rows = self.actual_rows
        main_height = 1 / (
            n_rows * (subfig_height * no_subfig + 1)
        )  # deduced by constraining the height proportion (total height = 1)
        sub_height = main_height * subfig_height
        row_heights = ([main_height] + [sub_height] * no_subfig) * n_rows
        specs = [[{"secondary_y": True}] * n_cols] * actual_rows

        fig = self.plotobj.create_plotly_figure(
            actual_rows,
            n_cols,
            height,
            width,
            specs=specs,
            row_heights=row_heights,
            shared_xaxes=True,
        )
        return go.Figure(fig)

    @staticmethod
    def process_data_df(
        data_df: pd.DataFrame,
        market: str = "US",
        lookback_years: int | None = None,
    ) -> pd.DataFrame:
        logger.validate(
            str(market).upper() in {"US", "A", "HK", "NA"},
            "market must be one of {'US', 'A', 'HK', 'NA'}",
        )
        if data_df.index.name == "time":
            out = data_df.copy()
            if lookback_years is not None and len(out.index) > 0:
                idx = pd.to_datetime(out.index, errors="coerce")
                latest = idx.max()
                if pd.notna(latest):
                    cutoff = latest - pd.DateOffset(years=lookback_years)
                    out = out[idx >= cutoff]
            try:
                out = out.drop(columns=["time"])
            except KeyError:
                pass
            return out
        # filter out weekends and market-specific holidays
        out = filter_trading_days(data_df, str(market).upper())
        out = out.set_index(pd.DatetimeIndex(out["time"]))
        if lookback_years is not None and len(out.index) > 0:
            latest = out.index.max()
            cutoff = latest - pd.DateOffset(years=lookback_years)
            out = out[out.index >= cutoff]
        out = out.drop(columns=["time"])
        return out

    def add_candle_trace(
        self,
        row: int = 0,
        col: int = 0,
        data_df: pd.DataFrame | None = None,
        x_arr: Sequence[Any] | None = None,
        open_arr: Sequence[float] | None = None,
        high_arr: Sequence[float] | None = None,
        low_arr: Sequence[float] | None = None,
        close_arr: Sequence[float] | None = None,
        market="NA",
        lookback_years: int | None = None,
        plot_spec: dict[str, Any] | None = None,
    ) -> None:
        if plot_spec is not None:
            candle_spec: dict[str, Any] = DEFAULT_CANDLE_SPEC.copy()
            candle_spec.update(plot_spec)
        else:
            candle_spec: dict[str, Any] = DEFAULT_CANDLE_SPEC.copy()
        logger.validate(
            data_df is not None
            or (
                x_arr is not None
                and open_arr is not None
                and high_arr is not None
                and low_arr is not None
                and close_arr is not None
            ),
            "data_df or x, open, high, low, close must be provided",
        )
        if data_df is None:
            data_df = pd.DataFrame(
                {
                    "time": x_arr,
                    "open": open_arr,
                    "high": high_arr,
                    "low": low_arr,
                    "close": close_arr,
                }
            )

        effective_lookback = self.lookback_years if lookback_years is None else lookback_years
        data_df = self.process_data_df(data_df, market, lookback_years=effective_lookback)
        self.markets[row, col] = str(market).upper()
        # filter out weekends and market-specific holidays
        self.data_dfs[row, col] = data_df
        self._update_xrangebreaks(row, col)
        self.fig.add_trace(
            go.Candlestick(
                x=data_df.index if "time" not in data_df.columns else data_df["time"],
                open=data_df["open"],
                high=data_df["high"],
                low=data_df["low"],
                close=data_df["close"],
                name=f"candle-{row}-{col}",
                **candle_spec,
            ),
            row=row * self.sub_rows + 1,
            col=col + 1,
        )
        self.plot_datas[row, col]["candle"] = self.fig.data[-1]

        if "vol" in self.data_dfs[row, col].columns:
            self.add_volume_trace(row, col)

    def add_volume_trace(
        self,
        row: int = 0,
        col: int = 0,
        *,
        x_arr: Sequence[Any] | None = None,
        volume_arr: Sequence[float] | None = None,
        plot_spec: dict[str, Any] | None = None,
        volume_def: Literal["volume", "amount"] = VOLUME_DEF,
    ) -> None:
        if plot_spec is not None:
            vol_spec: dict[str, Any] = DEFAULT_BAR_SPEC.copy()
            vol_spec.update(plot_spec)
        else:
            vol_spec: dict[str, Any] = DEFAULT_BAR_SPEC.copy()
        # Ensure volume is plotted on secondary y-axis
        if isinstance(self.data_dfs[row, col], pd.DataFrame):
            # Build a DataFrame from x_arr and volume_arr, then merge on "time"
            if volume_arr is None:
                logger.debug(
                    "volume_arr is None, use volume from candle df, so x_arr will not be used"
                )
                x_arr = None  # make sure x_arr is not used
            else:
                if x_arr is None:
                    logger.validate(
                        len(self.data_dfs[row, col]) == len(volume_arr),
                        "volume_arr must have the same length as candle df",
                    )
                    x_arr = (
                        self.data_dfs[row, col].index
                        if "time" not in self.data_dfs[row, col].columns
                        else self.data_dfs[row, col]["time"]
                    )
                volume_df = pd.DataFrame({"time": x_arr, "vol": volume_arr})
                volume_df["time"] = pd.to_datetime(volume_df["time"], errors="coerce")
                self.data_dfs[row, col] = pd.merge(
                    self.data_dfs[row, col],
                    volume_df,
                    on="time",
                    how="left",
                    suffixes=("", ""),
                )  # only leave "vol" that contained in candle df
            data_df = self.data_dfs[row, col]
        else:
            logger.info("no candle df found, so only plot volume")
            data_df = pd.DataFrame(
                {
                    "time": x_arr,
                    volume_def: volume_arr,
                    "open": [0] * len(x_arr),
                    "close": [0] * len(x_arr),
                }
            )
        vol_max = data_df[volume_def].max()

        self.fig.add_trace(
            go.Bar(
                x=data_df.index if "time" not in data_df.columns else data_df["time"],
                y=data_df[volume_def],
                marker_color=[
                    "#d41c4d" if o <= c else "#007f9b"
                    for o, c in zip(
                        self.data_dfs[row, col].get("open", []),
                        self.data_dfs[row, col].get("close", []),
                        strict=False,
                    )
                ]
                if "open" in self.data_dfs[row, col] and "close" in self.data_dfs[row, col]
                else "#b0b0b0",
                name=f"vol-{row}-{col}",
                **vol_spec,
            ),
            row=row * self.sub_rows + 1,
            col=col + 1,
            secondary_y=True,
        )
        self.fig.update_yaxes(
            range=[0, vol_max * 3],
            secondary_y=True,
            row=row * self.sub_rows + 1,
            col=col + 1,
        )
        self.plot_datas[row, col]["vol"] = self.fig.data[-1]

    def add_scatter_trace(
        self,
        row: int = 0,
        col: int = 0,
        label: str = "custom",
        position: int = 0,
        *,
        secondary_y: bool = False,
        x_arr: Sequence[Any] | None = None,
        y_arr: Sequence[float],
        plot_spec: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            row: int, row index of the subplot.
            col: int, column index of the subplot.
            label: str, label of the trace.
            position: int, position of the trace. 0 for main figure, i=1-N for the i-th subfigure.
        """
        if plot_spec is not None:
            scatter_spec: dict[str, Any] = DEFAULT_SCATTER_SPEC.copy()
            scatter_spec.update(plot_spec)
        else:
            scatter_spec: dict[str, Any] = DEFAULT_SCATTER_SPEC.copy()
        self.fig.add_trace(
            go.Scatter(
                x=x_arr
                if x_arr is not None
                else self.data_dfs[row, col].index
                if "time" not in self.data_dfs[row, col].columns
                else self.data_dfs[row, col]["time"],
                y=y_arr,
                name=f"{label}-{row}-{col}",
                hoverinfo="none",
                **scatter_spec,
            ),
            row=row * self.sub_rows + 1 + position,
            col=col + 1,
            secondary_y=secondary_y,
        )
        self.plot_datas[row, col][label] = self.fig.data[-1]


    def add_marker_trace(
        self,
        row: int = 0,
        col: int = 0,
        label: str = "custom",
        position: int = 0,
        *,
        secondary_y: bool = False,
        x_arr: Sequence[Any],
        y_arr: Sequence[float],
        type: Literal["markers", "text"] = "markers",
        marker: dict[str, Any] | None = None,
        text: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            row: int, row index of the subplot.
            col: int, column index of the subplot.
            label: str, label of the trace.
            position: int, position of the trace. 0 for main figure, i=1-N for the i-th subfigure.
        """
        if type == "markers":
            logger.validate(marker is not None, "marker_spec must be provided")
            marker_spec = {
                "symbol": marker,
                "color": "#9b1b30",
                "size": 14,
                "line": {
                    "color": "#0a6f69",
                    "width": 1,
                },
            }
            marker_spec.update(marker)
            self.fig.add_trace(
                go.Scatter(
                    x=x_arr,
                    y=y_arr,
                    mode="markers",
                    name=f"{label}-{row}-{col}",
                    marker=marker_spec,
                    showlegend=False,
                ),
                row=row * self.sub_rows + 1 + position,
                col=col + 1,
                secondary_y=secondary_y,
            )
        elif type == "text":
            logger.validate(text is not None, "text must be provided")
            logger.validate(len(x_arr) == len(text), "x_arr and text must have the same length")
            textfont = {
                "color": "#9b1b30",
                "size": 14,
                "family": "Arial",
                "weight": "bold",
            }
            self.fig.add_trace(
                go.Scatter(
                    x=x_arr,
                    y=y_arr,
                    mode="text",
                    name=f"{label}-{row}-{col}",
                    text=text,
                    textposition="top center",
                    textfont=textfont,
                    showlegend=False,
                    hoverinfo="none",
                ),
            )
        self.plot_datas[row, col][label] = self.fig.data[-1]

    def list_trace_names(self) -> list[str]:
        return [trace.name for trace in self.fig.data]

    def remove_trace(self, row: int = 0, col: int = 0, label: str = "custom") -> None:
        """
        Remove the trace with label from the plot and internal data structures.

        Args:
            row (int): Row index of the subplot.
            col (int): Column index of the subplot.
            label (str): Label of the trace to remove.
        """
        name = f"{label}-{row}-{col}"
        trace = self.plot_datas[row, col].get(label)
        if trace is not None:
            if trace.name != name:
                logger.warning(f"Trace name mismatch: {trace.name} != {name}")
                return
            # Remove the trace from the figure's data list
            try:
                del trace
                self.fig.data = [trace for trace in self.fig.data if trace.name != name]
                logger.info(f"Removed trace: {name}")
            except ValueError:
                pass  # Trace not found in fig.data, ignore
            # Remove from internal data structure

    def preset_main_indicators(self, row: int = 0, col: int = 0) -> None:
        df = self.data_dfs[row, col]
        df["avg1"] = (3 * df["close"] + df["high"] + df["low"] + df["open"]) / 6

        df["MA5"] = df["avg1"].rolling(5).mean()
        df["MA10"] = df["avg1"].rolling(10).mean()
        df["MA200"] = df["avg1"].rolling(200).mean()
        df["volMA5"] = df[VOLUME_DEF].rolling(5).mean()
        df["volMA120"] = df[VOLUME_DEF].rolling(120).mean()

        df["WMA8"] = ta.WMA(df["avg1"], timeperiod=8)
        df["high_wma"] = (
            df["WMA8"].rolling(2).max() + df["WMA8"].rolling(4).max() + df["WMA8"].rolling(8).max()
        ) / 3
        df["low_wma"] = (
            df["WMA8"].rolling(2).min() + df["WMA8"].rolling(4).min() + df["WMA8"].rolling(8).min()
        ) / 3

        df["draw_stop_fall"] = False
        condition_stop_fall = (df["low_wma"].shift(1) == df["WMA8"].shift(1)) & (df["WMA8"] > df["low_wma"])
        df.loc[condition_stop_fall, "draw_stop_fall"] = True

        df['high_smoothed'] = ta.EMA(ta.WMA(df['high'], timeperiod=20), timeperiod=90)
        df['low_smoothed'] = ta.EMA(ta.WMA(df['low'], timeperiod=20), timeperiod=90)
        df['width'] = df['high_smoothed'] - df['low_smoothed']
        df['channeltop_longperiod'] = df['high_smoothed'] + df['width'] * 2
        df['channelbot_longperiod'] = df['low_smoothed'] - df['width'] * 2

        # use XMA as a targeted result
        df['high_smoothed2'] = ta.EMA(ta.EMA(df['high'], timeperiod=25), timeperiod=25)
        df['low_smoothed2'] = ta.EMA(ta.EMA(df['low'], timeperiod=25), timeperiod=25)
        df['width2'] = df['high_smoothed2'] - df['low_smoothed2']
        df['channeltop_shortperiod'] = df['high_smoothed2'] + df['width2']
        df['channelbot_shortperiod'] = df['low_smoothed2'] - df['width2']


        df['positive'] = (df['channelbot_shortperiod'] >= df['channelbot_longperiod']) & (df['channeltop_shortperiod'] >= df['channeltop_longperiod'])
        df['negative'] = (df['channeltop_shortperiod'] <= df['channeltop_longperiod']) & (df['channelbot_shortperiod'] <= df['channelbot_longperiod'])
        df['neutral'] = (df['channelbot_shortperiod'] >= df['channelbot_longperiod']) & (df['channeltop_shortperiod'] <= df['channeltop_longperiod'])

        df['XA_17'] = df['low_smoothed2'] - 1.5 * df['width2']
        df['MID'] = (df['high_smoothed2'] + df['low_smoothed2']) / 2
        df['EMA_C_3'] = ta.EMA(df['close'], timeperiod=3)
        df['EMA_C_6'] = ta.EMA(df['close'], timeperiod=6)
        df['XA_19'] = ta.EMA(df['EMA_C_3'] - df['EMA_C_6'], timeperiod=9)
        df['XA_20'] = df['XA_19'].shift(1)

        df['EMA_C_3'] = ta.EMA(df['close'], timeperiod=3)
        df['EMA_C_9'] = ta.EMA(df['close'], timeperiod=9)
        df['P2_XA_21_EMA3'] = ta.EMA(df['EMA_C_3'] - df['EMA_C_9'], timeperiod=3)
        df['P2_XA_21_EMA9'] = ta.EMA(df['EMA_C_3'] - df['EMA_C_9'], timeperiod=9)
        df['XA_21'] = ta.EMA(df['P2_XA_21_EMA3'] - df['P2_XA_21_EMA9'], timeperiod=9)
        df['XA_22'] = df['XA_21'].shift(1)

        df['BUY1_SIGNAL'] = cross_over(df['channelbot_shortperiod'], df['low']) & df['positive']
        df['CAREFUL_SIGNAL_STRONG'] = cross_over(df['high'], df['channeltop_shortperiod']) & df['positive'] & (~ df['neutral'])
        df['SELL1_SIGNAL'] = cross_over(df['high'], df['channeltop_shortperiod']) & df['negative']
        df['CAREFUL_SIGNAL_WEAK'] = cross_over(df['channelbot_shortperiod'], df['low']) & df['negative'] & (~ df['neutral'])
        df['BUY2_SIGNAL'] = cross_over(df['channelbot_shortperiod'], df['low']) & df['neutral']
        df['SELL2_SIGNAL'] = cross_over(df['high'], df['channeltop_shortperiod']) & df['neutral']

        df['TEXT_BUY'] = np.where(df['BUY1_SIGNAL'] | df['BUY2_SIGNAL'], 'B', "")
        df['TEXT_BUY_Y'] = np.where(df['BUY1_SIGNAL'] | df['BUY2_SIGNAL'], df['low'], np.nan)

        df['TEXT_SELL'] = np.where(df['SELL1_SIGNAL'] | df['SELL2_SIGNAL'], 'S', "")
        df['TEXT_SELL_Y'] = np.where(df['SELL1_SIGNAL'] | df['SELL2_SIGNAL'], df['high'], np.nan)

        self.add_scatter_trace(row, col, "MA5", 0, y_arr=df["MA5"], plot_spec={"line": {"color": "#ffaec9", "width": 1, "dash": "solid"}})
        self.add_scatter_trace(row, col, "MA10", 0, y_arr=df["MA10"], plot_spec={"line": {"color": "#d01c1f", "width": 2, "dash": "solid"}})
        self.add_scatter_trace(row, col, "MA200", 0, y_arr=df["MA200"], plot_spec={"line": {"color": "#13ffff", "width": 4, "dash": "dot"}})

        self.add_scatter_trace(row, col, "top_long", 0, y_arr=df["channeltop_longperiod"], plot_spec={"line": {"color": "#007d60", "width": 1, "dash": "solid"}})
        self.add_scatter_trace(row, col, "bot_long", 0, y_arr=df["channelbot_longperiod"], plot_spec={"line": {"color": "#007d60", "width": 1, "dash": "solid"}})
        self.add_scatter_trace(row, col, "top_short", 0, y_arr=df["channeltop_shortperiod"], plot_spec={"line": {"color": "#ada396", "width": 1, "dash": "dash"}})
        self.add_scatter_trace(row, col, "bot_short", 0, y_arr=df["channelbot_shortperiod"], plot_spec={"line": {"color": "#ada396", "width": 1, "dash": "dash"}})

        self.add_scatter_trace(row, col, "volMA5", 0, secondary_y=True, y_arr=df["volMA5"])
        self.add_scatter_trace(row, col, "volMA120", 0, secondary_y=True, y_arr=df["volMA120"])

        self.add_marker_trace(row, col, "TEXT_BUY", 0, x_arr=df.index, y_arr=df["TEXT_BUY_Y"], type="text", text=df["TEXT_BUY"])
        self.add_marker_trace(row, col, "TEXT_SELL", 0, x_arr=df.index, y_arr=df["TEXT_SELL_Y"], type="text", text=df["TEXT_SELL"])

        df = df.drop(columns=[col for col in df.columns if col.startswith(('MA_C_', 'TEMP_', 'WMA_', 'P1_', 'P2_', 'XMA_')) and col not in ['XA_19', 'XA_21']], errors='ignore')

if __name__ == "__main__":
    import omnifinan as of
    from omnifinan.visualize import StockFigure
    plotobj = DataManipulator(1, 1, 10)
    df = of.unified_api.get_price_df("sh000001")
    plotobj = StockFigure(1, 1, 1, width=1000, height=3000)
    plotobj.add_candle_trace(0, 0, data_df=df, market="A")
    plotobj.add_volume_trace(0, 0)
    plotobj.preset_main_indicators(0,0)
    plotobj.plotobj.create_dash(plotobj.fig, port=11733, browser_open=True)
