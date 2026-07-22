from pathlib import Path
from ydata_profiling import ProfileReport

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "data_report.html"


def generate_report(df):

    try:

        profile = ProfileReport(
            df,
            title="AutoML Data Profiling Report",
            explorative=True,
            minimal=True,
        )

        profile.to_file(REPORT_PATH)

        return str(REPORT_PATH)

    except Exception as e:

        print(f"Profiling failed: {e}")

        return None