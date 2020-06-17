import os
from dataclasses import dataclass
from typing import List, Union

from .env import Settings


@dataclass
class Project:
    """project structure
    """
    output_root: str
    raw: str
    processed: str
    cache: str
    visualize: str

    def directories(self) -> List[str]:
        """
        return all directories.

        Returns:
            List of directories
        """
        return [
            self.output_root,
            self.raw,
            self.processed,
            self.cache,
            self.visualize
        ]


def setup_project(root: Union[str, None] = None, create: bool = True) -> Project:
    """
    setup new project directories which deal dataset and save model weights

    Args:
        root:
            project root path.
            By default, if `vivid.env.Settings.PROJECT_ROOT` is not none, use it.
            and use `~/vivid`
        create:
            create directories or not.
    Returns:
        project structure
    """

    if root is None:
        if Settings.PROJECT_ROOT:
            root = Settings.PROJECT_ROOT
        else:
            root = os.path.expanduser('~/vivid')
    else:
        root = os.path.abspath(root)
    project = Project(
        output_root=root,
        raw=os.path.join(root, 'raw'),
        processed=os.path.join(root, 'processed'),
        cache=os.path.join(root, 'cache'),
        visualize=os.path.join(root, 'visualize')
    )

    if create:
        for d in project.directories():
            os.makedirs(d, exist_ok=True)

    return project
