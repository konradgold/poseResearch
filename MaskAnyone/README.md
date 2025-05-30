# MaskAnyone

This is cloned from [MaskAnyone](https://github.com/MaskAnyone/MaskAnyoneProdInfrastructure.git).

## Setup

1. Enter AIIS server
2. Follow setup according to Section [Setup (local)](#setup-local)
3. Make sure the frontend runs on port 443 and is not blocked by a different application on that port.
4. Leave terminal open.
5. Open second terminal.
6. SSH on AIIS server with `ssh -L 8443:localhost:443 <AIIS server>`
7. Access frontend on [port 8443](https://localhost:8443) (https)

- Production Setup
  This document describes the process of setting up a production environment for MaskAnyone.

## Prerequisites

Make sure you have installed [Docker](https://docs.docker.com/get-docker/) on your system and set the appropriate permissions.
Furthermore, please also ensure that you have sufficient hard drive space available for the setup.
We recommend at least 50GB of free space.

## Setup

There are two different setup options.

If you want to run MaskAnyone locally on your computer, please use the "Setup (Local)" option.
The functionality is generally the same as with the server setup (see README of original repo), but you won't have a login and authentication service as you're the only person using it.
This reduces complexity and setup duration a bit.

> Please note the different docker-compose-<..>.yml files
>
> - `docker-compose-local.yml` comes without any GPU-support, thus not including SAM2 and OpenPose (MediaPipe-based masking only).
> - `docker-compose-local-gpu.yml` includes SAM2 and OpenPose (SAM2+MediaPipe/OpenPose-based masking).

## Setup (Local)

**Step 1: Open a terminal on your computer.**
This setup assumes that you're using a Linux(i.e. Debian)-based system.
However, generally speaking, Mask Anyone should also run on any other common operating system by following this setup.

**Step 2: Pull this repository.**
Run the following commands one by one to pull the repository.

```bash
git clone https://github.com/MaskAnyone/MaskAnyoneProdInfrastructure.git maskanyone
cd maskanyone
```

**Step 3: Configure the environment variables.**
Now we need to set up the environment variables. First run the following command:

```bash
cp .env.dist .env
```

Now open the newly created `.env` file (e.g. using `nano .env`) and do the following adjustments:

- Replace `<your-strong-password-1>` with as password of your choice. This is the password used to access the database.
- (Optional) Configure the number of workers you want as well as their available resources. This is very dependent on the hardware resources of your computer and can be flexibly adjusted later on.

**Step 4: Pull the infrastructure.**
Run the following command to pull all the images and prepare the application infrastructure:

```bash
docker compose -f docker-compose-local.yml pull
```

**Step 5: Start the application for the first time.**
To do so first start the database container.

```bash
docker compose -f docker-compose-local.yml up -d postgres
```

Then wait for 10 seconds to give it some time to prepare the database.
Afterward, also start the other containers.

```bash
docker compose -f docker-compose-local.yml up -d
```

> If you encounter an error like `Cannot start service proxy: driver failed programming external connectivity on endpoint mask-anyone-prod-test_proxy_1`, this likely means that another application is occupying the port 443. Stop this application and try again.

**Step 6: Verify that MaskAnyone is running.**
First check that all containers are up and running:

```bash
docker compose -f docker-compose-local.yml ps
```

If your containers are running, then please try accessing MaskAnyone at [port 8443](https://localhost:8433).
