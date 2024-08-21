import traceback


class Hyper:
    def __init__(self, *args):
        print(*args)


def load_hyper_from_files(filelist: list):
    hypers = []
    for filename in filelist:
        try:
            flag = False
            print(filename)
            hyper = Hyper()
            query_limit = 500
            a = filename.split("__model@")[1]
            model_name = a.split("__dataset@")[0]
            c = a.split("__dataset@")[1]
            dataset_name = c.split(".out")[0]
            if "__purb_limit_per_target@" in filename:
                e = filename.split("__purb_limit_per_target@")[1]
                purb_limit_per_target = int(e.split("__model")[0])
                setattr(hyper, "purb_limit_per_target", purb_limit_per_target)
            elif "attack_rate@" in filename:
                e = filename.split("attack_rate@")[1]
                attack_rate = float(e.split("__model")[0])
                setattr(hyper, "attack_rate", attack_rate)
            setattr(hyper, "model", model_name)
            setattr(hyper, "dataset", dataset_name)
            setattr(hyper, "query_limit", query_limit)

            file_object = open(filename, "r")
            lines = file_object.readlines()
            for line in lines:
                if "Params:" in line:
                    flag = True
                if flag and (": " in line):
                    item_ = line.strip()
                    res = item_.split(": ")
                    if len(res) > 1:
                        name = res[0]
                        val = res[1]
                        if "." in val:
                            val = float(val)
                        else:
                            val = int(val)
                        setattr(hyper, name, val)

            setattr(
                hyper, "level_2_query_limit", int(hyper.query_limit * hyper.query_radio)
            )
            setattr(
                hyper,
                "level_3_query_limit",
                int(hyper.query_limit * (1 - hyper.query_radio)),
            )
            hypers.append(hyper)
        except Exception as inst:
            traceback.print_exc()
            print(type(inst))  # the exception type
            print(inst.args)  # arguments stored in .args
            print(inst)  # __str__ allows args to be printed directly,
            continue
    return hypers
