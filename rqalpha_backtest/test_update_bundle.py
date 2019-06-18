import os


def test_update_bundle():
    # os.system("activate python3.5")
    # os.system("rqalpha update-bundle")
    import rqalpha.utils.bundle_helper
    rqalpha.utils.bundle_helper.update_bundle(data_bundle_path='D:\\rqalpha\data_bundle', locale="zh_Hans_CN")


if __name__ == '__main__':
    test_update_bundle()
