from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from langchain_core.runnables.graph import MermaidDrawMethod
from pyomnix.data_process import DataManipulator
from pyomnix.omnix_logger import get_logger
from pyomnix.utils import ObjectArray

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
    "width": 1000 * 60 * 60 * 24 * 0.8, # default using milliseconds
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
    ):
        self.actual_rows = (1 + no_subfig) * n_rows
        self.sub_rows = 1 + no_subfig  # rows of per group
        # the datas are stored w.r.t. subgroups not separate figures
        self.plotobj = DataManipulator(n_rows, n_cols, data_fill_value={"candle": [], "vol": []})
        self.markets = np.empty((n_rows, n_cols), dtype=str)
        self.plot_datas = self.plotobj.datas  # just an ObjectArray used for real-time plotting
        self.data_dfs = ObjectArray(n_rows, n_cols)  # used for data calculation
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
            width,
            height,
            specs=specs,
            row_heights=row_heights,
            shared_xaxes=True,
        )
        return go.Figure(fig)

    @staticmethod
    def process_data_df(data_df: pd.DataFrame, market: str = "US") -> pd.DataFrame:
        logger.validate(
            str(market).upper() in {"US", "A", "HK", "NA"},
            "market must be one of {'US', 'A', 'HK', 'NA'}",
        )
        if data_df.index.name == "time":
            try:
                data_df = data_df.drop(columns=["time"])
            except KeyError:
                pass
            return data_df
        # filter out weekends and market-specific holidays
        data_df = filter_trading_days(data_df, str(market).upper())
        data_df = data_df.set_index(pd.DatetimeIndex(data_df["time"]))
        data_df = data_df.drop(columns=["time"])
        return data_df

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

        data_df = self.process_data_df(data_df, market)
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
        volume_def: Literal["volume", "amount"] = "amount",
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
            range=[0, vol_max*3],
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
                **scatter_spec,
            ),
            row=row * self.sub_rows + 1 + position,
            col=col + 1,
            secondary_y=secondary_y,
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
        self.add_scatter_trace(
            row,
            col,
            "MA5",
            x_arr=self.data_dfs[row, col].index
            if "time" not in self.data_dfs[row, col].columns
            else self.data_dfs[row, col]["time"],
            y_arr=self.data_dfs[row, col]["MA5"],
        )
        self.add_scatter_trace(
            row,
            col,
            "MA10",
            x_arr=self.data_dfs[row, col].index
            if "time" not in self.data_dfs[row, col].columns
            else self.data_dfs[row, col]["time"],
            y_arr=self.data_dfs[row, col]["MA10"],
        )
        self.add_scatter_trace(
            row,
            col,
            "MA20",
            x_arr=self.data_dfs[row, col].index
            if "time" not in self.data_dfs[row, col].columns
            else self.data_dfs[row, col]["time"],
            y_arr=self.data_dfs[row, col]["MA20"],
        )
        self.add_scatter_trace(
            row,
            col,
            "MA50",
            x_arr=self.data_dfs[row, col].index
            if "time" not in self.data_dfs[row, col].columns
            else self.data_dfs[row, col]["time"],
            y_arr=self.data_dfs[row, col]["MA50"],
        )
        self.add_scatter_trace(
            row,
            col,
            "MA200",
            x_arr=self.data_dfs[row, col].index
            if "time" not in self.data_dfs[row, col].columns
            else self.data_dfs[row, col]["time"],
            y_arr=self.data_dfs[row, col]["MA200"],
        )
