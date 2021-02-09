from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager

import orjson as json
from tqdm.autonotebook import tqdm


def load_file(path, collection_foreign_id=None, n_entities=None, line_limit=4096):
    with tqdm(
        total=path.stat().st_size,
        desc=collection_foreign_id,
        leave=False,
        unit_divisor=1024,
        unit="B",
        unit_scale=True,
    ) as pbar:
        N = 0
        with open(path, "rb") as fd:
            for line in fd:
                pbar.update(fd.tell() - pbar.n)
                if len(line) > line_limit:
                    continue
                data = json.loads(line)
                if collection_foreign_id:
                    data["collection_foreign_id"] = collection_foreign_id
                N += 1
                yield data
                if n_entities and N >= n_entities:
                    return


def load_file_to_queue(queue, *args):
    for item in load_file(*args):
        queue.put(item)
    queue.put(None)


def entity_generator(data_paths, entities_per_file=None):
    data_paths.sort(key=lambda p: p.stat().st_size)
    total = sum(p.stat().st_size for p in data_paths)
    with tqdm(
        total=total, desc="Exports", unit_divisor=1024, unit="B", unit_scale=True
    ) as pbar:
        for path in data_paths:
            path = Path(path)
            cfid = path.parent.name
            entities = load_file(path, cfid, n_entities=entities_per_file)
            yield from entities
            pbar.update(path.stat().st_size)


def entity_generator_parallel(
    data_paths, queue_size=128, n_workers=None, entities_per_file=None
):
    n_workers = n_workers or 2 * cpu_count()
    with Manager() as manager:
        queue = manager.Queue(queue_size)
        with Pool(
            processes=n_workers, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)
        ) as pool:
            result = pool.starmap_async(
                load_file_to_queue,
                (
                    (queue, path, path.parent.name, entities_per_file)
                    for path in data_paths
                ),
            )
            with tqdm(total=len(data_paths), desc="Exports") as pbar:
                while not (queue.empty() and result.ready()):
                    item = queue.get()
                    if item is not None:
                        yield item
                    else:
                        pbar.update(1)
