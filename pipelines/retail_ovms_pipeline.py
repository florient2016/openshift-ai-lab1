
from typing import Optional
import os

# KFP v2 SDK
from kfp import dsl
from kfp import compiler

# ----------
# Components
# ----------

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "scikit-learn==1.4.2",
        "joblib==1.3.2",
        "pandas==2.2.1",
        "numpy==1.26.4"
    ],
)
def train_retail_model(
    n_estimators: int,
    max_depth: int,
    joblib_out: dsl.OutputPath("Model") # type: ignore
):
    """Train a simple RandomForestRegressor on a synthetic retail-like dataset
    and save as joblib for downstream steps. In real life, replace with your data prep.
    """
    import numpy as np, pandas as pd # pyright: ignore[reportMissingModuleSource]
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import joblib, json

    # Synthetic feature table: [promo_flag, dow, month, store_id, sku_id, price, lag_1, lag_7, holiday_flag, trend]
    rng = np.random.default_rng(42)
    n = 2000
    X = pd.DataFrame({
        "promo": rng.integers(0, 2, n),
        "dow": rng.integers(0, 7, n),
        "month": rng.integers(1, 13, n),
        "store_id": rng.integers(1, 51, n),
        "sku_id": rng.integers(1, 501, n),
        "price": rng.normal(10, 2, n).clip(1, None),
        "lag_1": rng.normal(100, 30, n),
        "lag_7": rng.normal(100, 30, n),
        "holiday": rng.integers(0, 2, n),
        "trend": np.linspace(0, 1, n),
    })
    # Create a target depending on features + noise
    y = (
        2.0*X["promo"] - 0.3*X["price"] + 0.1*X["lag_1"] + 0.05*X["lag_7"]
        + 0.5*X["holiday"] + 5*X["trend"] + rng.normal(0, 3, n)
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    meta = {"mae": mae, "n_features_in_": int(getattr(rf, "n_features_in_", X.shape[1]))}
    print("Train MAE:", mae)

    obj = {"model": rf, "meta": meta, "feature_names": list(X.columns)}
    import joblib as _joblib
    _joblib.dump(obj, joblib_out)
    print("Saved joblib to", joblib_out)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["minio==7.2.7"]
)
def upload_joblib_to_minio(
    joblib_in: dsl.InputPath("Model"), # pyright: ignore[reportInvalidTypeForm]
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    s3_prefix: str,
    minio_secure: bool = False,
    uploaded_uri_out: dsl.OutputPath(str) = "uploaded_uri.txt", # type: ignore
):
    from minio import Minio
    from pathlib import Path

    client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=bool(minio_secure))
    if not client.bucket_exists(minio_bucket):
        client.make_bucket(minio_bucket)
        print(f"Created bucket {minio_bucket}")

    key = f"{s3_prefix}/retail_sales_model.joblib"
    client.fput_object(minio_bucket, key, joblib_in, content_type="application/octet-stream")
    s3_uri = f"s3://{minio_bucket}/{key}"
    Path(uploaded_uri_out).write_text(s3_uri)
    print("Uploaded:", s3_uri)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["scikit-learn==1.4.2", "joblib==1.3.2", "skl2onnx==1.16.0", "onnx==1.16.0"]
)
def convert_joblib_to_onnx(
    joblib_in: dsl.InputPath("Model"), # type: ignore
    feature_count: int,
    onnx_dir_out: dsl.OutputPath("Directory") # type: ignore
):
    import joblib, os
    from skl2onnx import convert_sklearn # type: ignore
    from skl2onnx.common.data_types import FloatTensorType # type: ignore
    from pathlib import Path

    loaded = joblib.load(joblib_in)
    pipeline = loaded.get("model") if isinstance(loaded, dict) and "model" in loaded else loaded
    n_features = int(getattr(pipeline, "n_features_in_", feature_count))

    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_types)

    out_dir = Path(onnx_dir_out) / "1"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Saved ONNX at", onnx_path)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["minio==7.2.7"]
)
def upload_onnx_to_minio(
    onnx_dir_in: dsl.InputPath("Directory"), # type: ignore
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    s3_prefix: str,
    minio_secure: bool = False,
    #ovms_root_out: dsl.OutputPath(str) = "ovms_root.txt"
    ovms_root_out: dsl.OutputPath() = "ovms_root.txt" # type: ignore
):
    from minio import Minio
    from pathlib import Path
    import os

    client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=bool(minio_secure))
    if not client.bucket_exists(minio_bucket):
        client.make_bucket(minio_bucket)

    # Upload directory recursively to s3://bucket/<prefix>/openvino/
    root = Path(onnx_dir_in)
    ovms_prefix = f"{s3_prefix}/openvino"
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root)
            key = f"{ovms_prefix}/{rel.as_posix()}"
            client.fput_object(minio_bucket, key, str(p), content_type="application/octet-stream")
            print("Uploaded", key)

    s3_root = f"s3://{minio_bucket}/{ovms_prefix}"
    Path(ovms_root_out).write_text(s3_root)
    print("OVMS model root:", s3_root)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes==28.1.0", "requests==2.31.0", "pyyaml==6.0.1", "kfp[kubernetes]==2.9.0" ]
)
def deploy_ovms_and_test(
    model_name: str,
    ovms_model_root_s3: dsl.InputPath(), # type: ignore
    #ovms_model_root_s3: dsl.InputPath(str), 
    #ovms_model_root_s3: str, # type: ignore
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_secure: bool,
    n_features: int,
    isvc_url_out: dsl.OutputPath(str) = "isvc_url.txt", # type: ignore
):
    """Create Secret, ServiceAccount, and InferenceService for OVMS. Wait Ready and test inference."""
    import time, json, requests, os
    from kubernetes import client, config
    from pathlib import Path

    endpoint_url = ("https://" if minio_secure else "http://") + minio_endpoint
    #model_root = Path(ovms_model_root_s3).read_text().strip()
    if os.path.exists(ovms_model_root_s3):
        model_root = Path(ovms_model_root_s3).read_text().strip()
    else:
        model_root = ovms_model_root_s3

    # Load kube config (in-cluster or local)
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    core = client.CoreV1Api()
    custom = client.CustomObjectsApi()

    # Namespace detection
    ns = "default"
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
            ns = f.read().strip()
    except Exception:
        pass

    # Create/patch Secret
    secret_name = "s3-credentials"
    secret_body = client.V1Secret(
        metadata=client.V1ObjectMeta(name=secret_name),
        type="Opaque",
        string_data={
            "AWS_ACCESS_KEY_ID": minio_access_key,
            "AWS_SECRET_ACCESS_KEY": minio_secret_key,
            "AWS_ENDPOINT_URL": endpoint_url,
        },
    )
    try:
        core.create_namespaced_secret(ns, secret_body)
        print("Created Secret", secret_name)
    except client.exceptions.ApiException as e:
        if e.status == 409:
            core.patch_namespaced_secret(secret_name, ns, secret_body)
            print("Patched existing Secret", secret_name)
        else:
            raise

    # ServiceAccount
    sa_name = "minio-s3-sa"
    sa_body = client.V1ServiceAccount(
        metadata=client.V1ObjectMeta(name=sa_name),
        secrets=[client.V1ObjectReference(name=secret_name)]
    )
    try:
        core.create_namespaced_service_account(ns, sa_body)
        print("Created ServiceAccount", sa_name)
    except client.exceptions.ApiException as e:
        if e.status == 409:
            core.patch_namespaced_service_account(sa_name, ns, sa_body)
            print("Patched existing ServiceAccount", sa_name)
        else:
            raise

    # InferenceService spec
    isvc_name = f"{model_name}-ovms"
    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {"name": isvc_name},
        "spec": {
            "predictor": {
                "model": {
                    "runtime": "ovms",
                    "protocolVersion": "v2",
                    "storageUri": model_root
                },
                "serviceAccountName": sa_name
            }
        }
    }

    group = "serving.kserve.io"
    version = "v1beta1"
    plural = "inferenceservices"

    try:
        custom.create_namespaced_custom_object(group, version, ns, plural, isvc)
        print("Created InferenceService", isvc_name)
    except client.exceptions.ApiException as e:
        if e.status == 409:
            custom.patch_namespaced_custom_object(group, version, ns, plural, isvc_name, isvc)
            print("Patched existing InferenceService", isvc_name)
        else:
            raise

    # Wait for Ready
    def get_status():
        obj = custom.get_namespaced_custom_object(group, version, ns, plural, isvc_name)
        return obj.get("status", {})

    url = None
    print("Waiting for InferenceService to be Ready...")
    for _ in range(60):
        st = get_status()
        conds = st.get("conditions", [])
        if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds):
            url = st.get("url")
            break
        time.sleep(10)

    if not url:
        raise RuntimeError("InferenceService not Ready in time. Check events/logs.")

    Path(isvc_url_out).write_text(url)
    print("Service URL:", url)

    # Probe metadata then infer
    name = isvc_name
    meta = requests.get(f"{url}/v2/models/{name}", timeout=30)
    if meta.status_code != 200:
        # try model_name
        alt = model_name
        meta = requests.get(f"{url}/v2/models/{alt}", timeout=30)
        if meta.ok:
            name = alt

    input_name = "input"
    datatype = "FP32"
    shape = [1, n_features]
    try:
        j = meta.json()
        if "inputs" in j and j["inputs"]:
            input_name = j["inputs"][0].get("name", input_name)
            datatype = j["inputs"][0].get("datatype", datatype)
            shape = j["inputs"][0].get("shape", shape)
    except Exception:
        pass

    dummy = [0.0] * (shape[-1] if isinstance(shape, list) and len(shape) > 0 else n_features)
    payload = {
        "inputs": [{
            "name": input_name,
            "shape": shape,
            "datatype": datatype,
            "data": dummy
        }]
    }
    r = requests.post(f"{url}/v2/models/{name}/infer", json=payload, timeout=60)
    print("Inference status:", r.status_code)
    print("Inference response:", r.text[:1000])


# -------------
# The Pipeline
# -------------

@dsl.pipeline(
    name="retail-ovms-end2end",
    description="Train -> Upload to MinIO -> Convert to ONNX -> Upload -> Deploy OVMS -> Test"
)
def retail_ovms_pipeline(
    # Training params
    n_estimators: int = 150,
    max_depth: int = 12,

    # MinIO params
    minio_endpoint: str = "<CHANGE_ME>",
    minio_access_key: str = "<CHANGE_ME>",
    minio_secret_key: str = "<CHANGE_ME>",
    minio_bucket: str = "artifacts",
    s3_prefix: str = "retail",
    minio_secure: bool = False,

    # Serving params
    model_name: str = "retail-sales",
    feature_count: int = 10
):
    train = train_retail_model(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    up_joblib = upload_joblib_to_minio(
        joblib_in=train.outputs["joblib_out"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
        s3_prefix=s3_prefix,
        minio_secure=minio_secure
    )

    to_onnx = convert_joblib_to_onnx(
        joblib_in=train.outputs["joblib_out"],
        feature_count=feature_count
    )

    up_onnx = upload_onnx_to_minio(
        onnx_dir_in=to_onnx.outputs["onnx_dir_out"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
        s3_prefix=s3_prefix,
        minio_secure=minio_secure
    )

    deploy = deploy_ovms_and_test(
        model_name=model_name,
        ovms_model_root_s3=up_onnx.outputs["ovms_root_out"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_secure=minio_secure,
        n_features=feature_count
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=retail_ovms_pipeline,
        package_path="retail_ovms_pipeline.yaml"
    )
    print("Compiled to retail_ovms_pipeline.yaml")
