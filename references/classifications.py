
oakland_classes = {
    1000: "to_label",
    1001: "undet",
    1002: "linear_misc",
    1003: "surf_misc",
    1004: "scatter_misc",
    1005: "artifact",
    1100: "default_wire",
    1101: "wire_bundle",
    1102: "isolated_wire",
    1103: "utility_pole",
    1104: "crossarm",
    1105: "support_wire",
    1106: "support_pole",
    1107: "lamp_support",
    1108: "transformer",
    1109: "fire_hydrant",
    1110: "post",
    1111: "sign",
    1112: "pylon",
    1113: "bench",
    1114: "lamp",
    1115: "traffic_lights",
    1116: "traffic_lights_support",
    1117: "garbate",
    1118: "crosswalk_light",
    1119: "parking_meter",
    1200: "load_bearing",
    1201: "cliff",
    1202: "ground",
    1203: "paved_road",
    1204: "trail",
    1205: "curb",
    1206: "walkway",
    1207: "guardrail",
    1300: "foliage",
    1301: "grass",
    1302: "small_trunk",
    1303: "large_trunk",
    1304: "thin_branch",
    1305: "thick_branch",
    1306: "shrub",
    1400: "facade",
    1401: "wall",
    1402: "stairs",
    1403: "door",
    1404: "window",
    1405: "chimney",
    1406: "roof",
    1407: "chainlinkfence",
    1408: "fence",
    1409: "gate",
    1410: "ceiling",
    1411: "facade_ledge",
    1412: "column",
    1413: "mailbox",
    1500: "human",
    1501: "vehicle",
    1600: "rock",
    1601: "concertina_wire",
}

class_update = {2000: [1400, 1401, 1410, 1411, 1412],
                 2001: [1101, 1102, 1105],
                 2002: [1103, 1104, 1107, 1110, 1112, 1114, 1115, 1116, 1119],
                 1202: [1202, 1203, 1205, 1206, 1301, 1402],
                 1300: [1300, 1301, 1302, 1303, 1305, 1306],
                 9999: [1001, 1002, 1003, 1108, 1109, 1111, 1113, 1117, 1500],
                 1501: [1501]}

class_decode = {2000: 'building',
                2001: 'wires',
                2002: 'poles',
                1202: 'ground',
                9999: 'other',
                1300: 'foliage',
                1501: 'vehicle'}

encode_class = {1400: 2000,
                 1401: 2000,
                 1410: 2000,
                 1411: 2000,
                 1412: 2000,
                 1101: 2001,
                 1102: 2001,
                 1105: 2001,
                 1103: 2002,
                 1104: 2002,
                 1107: 2002,
                 1110: 2002,
                 1112: 2002,
                 1114: 2002,
                 1115: 2002,
                 1116: 2002,
                 1119: 2002,
                 1202: 1202,
                 1203: 1202,
                 1205: 1202,
                 1206: 1202,
                 1301: 1300,
                 1402: 1202,
                 1300: 1300,
                 1302: 1300,
                 1303: 1300,
                 1305: 1300,
                 1306: 1300,
                 1001: 9999,
                 1002: 9999,
                 1003: 9999,
                 1108: 9999,
                 1109: 9999,
                 1113: 9999,
                 1117: 9999,
                 1500: 9999,
                 1111: 9999,
                 1501: 1501,
                 9999: 9999,
                1409: 9999,
                1200: 2000,
                1408: 9999,
                1413: 9999,
                1118: 9999}

class_name_to_id = {
    'building': 2000,
    'wires': 2001,
    'poles': 2002,
    'ground': 1202,
    'other': 9999,
    'foliage': 1300,
    'vehicle': 1501
}

# undet 18307 x
# linear_misc 769 x
# surf_misc 502 x
# wire_bundle 4970
# isolated_wire 2294
# utility_pole 5927
# crossarm 616
# support_wire 45
# lamp_support 340
# transformer 26 x
# fire_hydrant 86 x
# post 1209
# sign 2631
# pylon 8
# bench 384 x
# lamp 359
# traffic_lights 508
# traffic_lights_support 512
# garbate 64 x
# crosswalk_light 88
# parking_meter 69
# load_bearing 92888
# ground 3171
# paved_road 883676
# curb 138
# walkway 19966
# foliage
# grass 8368
# small_trunk 931
# large_trunk 2319
# thick_branch 284
# shrub 7991
# facade 95513
# wall 12443
# stairs 301
# fence 749
# gate 129
# ceiling 485
# facade_ledge 1116
# column 1822
# mailbox 155
# human 237 x
# vehicle 59183
