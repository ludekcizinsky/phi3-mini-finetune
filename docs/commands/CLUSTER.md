## Useful commands for scheduling jobs at the SLURM cluster

All experiments are being run from the [SCITAS](http://scitas.epfl.ch/) cluster at EPFL. To log onto the cluster

```bash
ssh <username>@izar.epfl.ch
```

*You can add SSH keys into `~/.ssh/authorized_keys` and specify the path to the private key while logging in to avoid re-entering the GASPAR password.*

The project repository should be cloned into the folder `/scratch/izar/<username>/mnlp-project`

```bash
cd /scratch/izar/<username>; git clone https://github.com/CS-552/project-m2-2024-mlp mnlp-project
```

Before running a job, you need to make sure you have all dependencies installed. To do so, you can run the following command

```bash
chmod +x setup.sh; ./setup.sh
```

*If you have already `git-lfs` installed, you can skip its installation by adding `--skip-git-lfs`*.

To run a job, modify the script to be run inside `run.sh` and then execute

```bash
sbatch run.sh
```

To interactively debug from a compute node (with GPU support) execute

```bash
Sinteract -a cs-552 -p debug -q debug -g gpu -m 32G -t 1:00:00
```
Or with the reservation (be aware that this will block other jobs):

```bash
Sinteract -a cs-552 -r cs-552 -q cs-552 -g gpu -m 32G -t 1:00:00
```

For more details, look at this [SCITAS Guide](https://docs.google.com/document/d/1Wby_mwsUhJ4E7J7M0u6GO9-2zNVJ6lU4n12KvEa34Vk/edit?usp=sharing) provided by the course staff or check out the official [SCITAS Documentation](https://scitas-doc.epfl.ch/) or [SLURM Documentation](http://slurm.schedmd.com/man_index.html).
