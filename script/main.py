import glob
import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import zipfile
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import fire
import fsspec
import GPUtil
import pandas as pd
from huggingface_hub import HfApi, HfFolder, login
from loguru import logger

import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str
from datasets import load_dataset



def handle_found_object(
        local_path: str,
        file_identifier: str,
        metadata: Dict[str, Any],
        num_renders: int,
        render_dir: str,
        only_northern_hemisphere: bool,
        gpu_devices: Union[int, List[int]],
        render_timeout: int,
) -> bool:
    """Called when an object is successfully found and downloaded.

    Here, the object has the same sha256 as the one that was downloaded with
    Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
    it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.
        successful_log_file (str): Name of the log file to save successful renders to.
        failed_log_file (str): Name of the log file to save failed renders to.

    Returns: True if the object was rendered successfully, False otherwise.
    """
    save_uid = get_uid_from_str(file_identifier)
    args = f"--object_path '{local_path}' --num_renders {num_renders}"

    # get the GPU to use for rendering
    using_gpu: bool = False
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        # get the target directory for the rendering job
        target_directory = os.path.join(temp_dir, save_uid)
        logger.info(f"Rendering object {file_identifier} to {target_directory}.")
        os.makedirs(target_directory, exist_ok=True)
        args += f" --output_dir '{target_directory}'"

        logger.info(f"Platform: {platform.system()}.")
        # check for Linux / Ubuntu or MacOS
        if platform.system() == "Linux" and using_gpu:
            args += " --engine BLENDER_EEVEE_NEXT"
        elif platform.system() == "Darwin" or (
                platform.system() == "Linux" and not using_gpu
        ):
            # As far as I know, MacOS does not support BLENER_EEVEE, which uses GPU
            # rendering. Generally, I'd only recommend using MacOS for debugging and
            # small rendering jobs, since CYCLES is much slower than BLENDER_EEVEE.
            args += " --engine BLENDER_EEVEE_NEXT"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # check if we should only render the northern hemisphere
        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"

        # get the command to run
        command = f"blender-4.2.3-linux-x64/blender --background --python blender_script.py -- {args}"
        if using_gpu:
            command = f"export DISPLAY=:0.{gpu_i} && {command}"

        # render the object (put in dev null)
        logger.info(f"Running command: {command}")
        subprocess.run(
            ["bash", "-c", command],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # check that the renders were saved successfully
        png_files = glob.glob(os.path.join(target_directory, "*.png"))
        metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        npy_files = glob.glob(os.path.join(target_directory, "*.npy"))
        if (
                (len(png_files) != num_renders)
                or (len(npy_files) != num_renders)
                or (len(metadata_files) != 1)
        ):
            logger.error(
                f"Found object {file_identifier} was not rendered successfully!"
            )
            return False

        # move the all the files to the render_dir
        fs, path = fsspec.core.url_to_fs(render_dir)
        logger.info(f"Moving files to {path}.")
        # move the files to the render directory with the save_uid
        fs.mv(target_directory, os.path.join(path, file_identifier), recursive=True)


        # create metadata file for hunging face
        create_metadata_hunging_face(os.path.join(path, file_identifier), metadata)


        return True

def create_metadata_hunging_face(local_path: str, metadata:Dict[str, Any]):
    # verify that metadata in rendir_exist
    metadata_file = os.path.join(local_path, 'metadata.jsonl')
    metadata_render = []

    # get all file in the local path
    for file in os.listdir(local_path):
        # get only npy and png files
        if file.endswith('.png'):
            # get the name of the file
            name = file.split('.')[0]
            # create metadata file
            npy_file = os.path.join(local_path, f"{name}.npy")
            if os.path.exists(npy_file):
                npy_path = npy_file
            else:
                raise FileNotFoundError(f"File {npy_file} not found.")

            metadata_render.append({
                "img_path": os.path.join(local_path, file),
                "npy_path": npy_path,
                "caption": metadata["caption"]
            })

            # save metadata file
            with open(metadata_file, 'w') as f:
                # order metadata_render by file_name
                metadata_render_sorted = sorted(metadata_render, key=lambda x: x['img_path']);
                for metadata in metadata_render_sorted:
                    json.dump(metadata, f, indent=4)
                    f.write("\n")







def get_objects(path: str) -> pd.DataFrame:
    """Returns a DataFrame of example objects to use for debugging."""
    return pd.read_json(path, orient="records")


def render_objects(
        render_dir: str = "~/EPITA/PFEE/obj-3D/renders/",
        path: str = "~/EPITA/PFEE/obj-3D/dataset_3D.json",
        download_dir: Optional[str] = None,
        num_renders: int = 1,
        processes: Optional[int] = None,
        save_repo_format: Optional[Literal["zip", "tar", "tar.gz", "files"]] = None,
        only_northern_hemisphere: bool = False,
        render_timeout: int = 300,
        gpu_devices: Optional[Union[int, List[int]]] = None,
) -> None:
    """Renders objects in the Objaverse-XL dataset with Blender

    Args:
        render_dir (str, optional): Directory where the objects will be rendered.
        download_dir (Optional[str], optional): Directory where the objects will be
            downloaded. If None, the objects will not be downloaded. Defaults to None.
        num_renders (int, optional): Number of renders to save of the object. Defaults
            to 12.
        processes (Optional[int], optional): Number of processes to use for downloading
            the objects. If None, defaults to multiprocessing.cpu_count() * 3. Defaults
            to None.
        save_repo_format (Optional[Literal["zip", "tar", "tar.gz", "files"]], optional):
            If not None, the GitHub repo will be deleted after rendering each object
            from it.
        only_northern_hemisphere (bool, optional): Only render the northern hemisphere
            of the object. Useful for rendering objects that are obtained from
            photogrammetry, since the southern hemisphere is often has holes. Defaults
            to False.
        render_timeout (int, optional): Number of seconds to wait for the rendering job
            to complete. Defaults to 300.
        gpu_devices (Optional[Union[int, List[int]]], optional): GPU device(s) to use
            for rendering. If an int, the GPU device will be randomly selected from 0 to
            gpu_devices - 1. If a list, the GPU device will be randomly selected from
            the list. If 0, the CPU will be used for rendering. If None, all available
            GPUs will be used. Defaults to None.

    Returns:
        None
    """
    if platform.system() not in ["Linux", "Darwin"]:
        raise NotImplementedError(
            f"Platform {platform.system()} is not supported. Use Linux or MacOS."
        )
    if download_dir is None and save_repo_format is not None:
        raise ValueError(
            f"If {save_repo_format=} is not None, {download_dir=} must be specified."
        )
    if download_dir is not None and save_repo_format is None:
        logger.warning(
            f"GitHub repos will not save. While {download_dir=} is specified, {save_repo_format=} None."
        )

    # get the gpu devices to use
    parsed_gpu_devices: Union[int, List[int]] = 0
    if gpu_devices is None:
        parsed_gpu_devices = len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")

    if processes is None:
        processes = multiprocessing.cpu_count() * 3

    # get the objects to render
    objects = get_objects(path)
    logger.info(f"Provided {len(objects)} objects to render.")
    for path, name, caption in objects[["path", "name", "caption"]].values:
        path = os.path.expanduser(path)
        handle_found_object(
            local_path=path,
            file_identifier=name,
            metadata={"caption": caption},
            num_renders=num_renders,
            render_dir=render_dir,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=parsed_gpu_devices,
            render_timeout=render_timeout,
        )

        token = "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"
        login(token)
        dataset = load_dataset("json",data_files="./dataset/Scalpel/metadata.jsonl", split='train')
        dataset = dataset.map(map_function)
        dataset = dataset.remove_columns(["img_path", "npy_path"])

        dataset.push_to_hub("Medical-3D/medical-3D")


def map_function(example):
    from PIL import Image
    import numpy as np

    # Open the image file using Pillow
    image = Image.open(example["img_path"])

    # Load the numpy file and flatten it
    npy_data = np.load(example["npy_path"]).flatten()

    return {
        "image": image,
        "npy_data": npy_data,
        "caption": example["caption"]
    }


if __name__ == "__main__":
    fire.Fire(render_objects)
