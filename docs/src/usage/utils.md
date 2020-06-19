# Utility

## プロジェクトディレクトリのセットアップ

機械学習プロジェクトを行なう際によく使うディレクトリを作る機能です。`vivid.setup.setup_project` を呼び出すとディレクトリの作成と、それらへのパスを取得することができます。プロジェクトのルートディレクトリを設定しない場合 `vivid.env.PROJECT_ROOT` が使用されます。これは環境変数 `VIVID_PROJECT_ROOT` か、これが設定されていない場合には `~/vivid` を使います。

```python
from vivi.setup import setup_project, Project

project_dirs = setup_project('path/to/project_root') # type: Project
```

## 関数のキャッシュ化

関数の呼び出し結果をキャッシュとして保存して、二回目以降呼び出された場合には実際には実行せずキャッシュを返す機能を提供しています。簡単な使い方は `cachable` をデコレータとして使用する方法です。

```python
from vivid.cachable import cachable

@cachable
def some_func():
  return [1, 2, 3]
```

デフォルトでは関数名がキャッシュディレクトリとして使用されますが、明示的に指定することもできます。

```python
@cachable('foo')
def some_func():
  return [1, 2, 3]
```

デフォルトでは、 `vivid.setup.setup_project` が作るプロジェクト instance の `cache` がキャッシュの保存ディレクトリとして使用されます。これは自分で指定することも可能です。

```python
@cachable('foo', directory='/my/cache/dir')
def some_func():
  return [1, 2, 3]
```

`cachable` は引数の解析機能が入っています。ハッシュ化可能なもの (たとえば int / string など。駄目な例は pd.DataFrame など) に限定されますが、引数によって結果が変わるような関数でもキャッシュ化することが可能です。

```python
@cachable('foo', directory='/my/cache/dir')
def some_func(x=10):
  return x

some_func()

some_func() # キャッシュが利用される
some_func(x=20) # 引数が変わったので新しくキャッシュファイルが作成される

some_func(x=20) # 二回目はキャッシュが使われる
some_func(x=10) # 同じ引数の時もキャッシュが使われる
```
