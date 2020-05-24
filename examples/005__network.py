import pandas as pd

from vivid.backends.experiments import LocalExperimentBackend
from vivid.core import BaseBlock


class TestBlock(BaseBlock):
    def _fit_core(self, source_df, y, experiment: LocalExperimentBackend) -> pd.DataFrame:
        print(experiment.output_dir, self.runtime_env)
        return source_df

    def transform(self, source_df):
        return source_df


if __name__ == '__main__':
    a = TestBlock('a')
    b = TestBlock('b')
    c = TestBlock('c')

    d = TestBlock('d', parent=[a, b, c])
    e = TestBlock('e', parent=[a, c])
    f = TestBlock('f', parent=[e, b, d])

    g = TestBlock('g', parent=[f, e, a, c, d])

    exp = LocalExperimentBackend(namespace='./outputs/test')
    input_df = pd.DataFrame()
    g.fit(input_df, experiment=exp)

    g.predict(input_df, experiment=exp)
