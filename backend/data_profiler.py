from ydata_profiling import ProfileReport


def generate_report(df):

    try:

        report = ProfileReport(
            df,
            explorative=True,
            minimal=True
        )

        report.to_file("data_report.html")

        print("EDA report generated")

    except Exception as e:

        print("EDA generation failed:", e)