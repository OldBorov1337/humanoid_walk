#!/usr/bin/env python3
import sys
import os
import models
from dm_control import mjcf

# ------------------------------------------------------------------
# 1) Путь к вашему jvrc1.xml внутри локального submodule jvrc_mj_description
# ------------------------------------------------------------------
JVRC_DESCRIPTION_PATH = os.path.join(
    os.path.dirname(models.__file__),
    "jvrc_mj_description", "xml", "jvrc1.xml"
)
assert os.path.exists(JVRC_DESCRIPTION_PATH), (
    f"Не нашёл jvrc1.xml по пути {JVRC_DESCRIPTION_PATH}"
)

# имена групп суставов, которые мы сохраняем
LEG_JOINTS = [
    'R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'R_KNEE', 'R_ANKLE_R', 'R_ANKLE_P',
    'L_HIP_P', 'L_HIP_R', 'L_HIP_Y', 'L_KNEE', 'L_ANKLE_R', 'L_ANKLE_P'
]
ARM_BODIES = {
    "R_SHOULDER_P_S":[0, -0.052, 0],
    "R_SHOULDER_R_S":[-0.17, 0, 0],
    "R_ELBOW_P_S":[0, -0.524, 0],
    "L_SHOULDER_P_S":[0, -0.052, 0],
    "L_SHOULDER_R_S":[ 0.17, 0, 0],
    "L_ELBOW_P_S":[0, -0.524, 0],
}

def builder(export_path, config):
    print("Modifying XML model...")
    # ------------------------------------------------------------------
    # Загрузка исходного XML
    # ------------------------------------------------------------------
    mjcf_model = mjcf.from_path(JVRC_DESCRIPTION_PATH)
    mjcf_model.model = 'jvrc'

    # ------------------------------------------------------------------
    # Общие правки: размер буферов, небо, статистика
    # ------------------------------------------------------------------
    mjcf_model.size.njmax   = -1
    mjcf_model.size.nconmax = -1
    mjcf_model.statistic.meansize = 0.1
    mjcf_model.statistic.meanmass = 2
    for tx in mjcf_model.asset.texture:
        if getattr(tx, 'type', None) == "skybox":
            tx.rgb1 = '1 1 1'
            tx.rgb2 = '1 1 1'

    # ------------------------------------------------------------------
    # Удаляем всё, что не ногам: контакты, моторы, «лишние» суставы
    # ------------------------------------------------------------------
    mjcf_model.contact.remove()
    for mot in list(mjcf_model.actuator.motor):
        if mot.joint.name not in LEG_JOINTS:
            mot.remove()
    for joint in (
        ['WAIST_Y','WAIST_P','WAIST_R'] +
        ['NECK_Y','NECK_R','NECK_P'] +
        ['R_UTHUMB','R_LTHUMB','R_UINDEX','R_LINDEX','R_ULITTLE','R_LLITTLE'] +
        ['L_UTHUMB','L_LTHUMB','L_UINDEX','L_LINDEX','L_ULITTLE','L_LLITTLE'] +
        ['R_SHOULDER_P','R_SHOULDER_R','R_SHOULDER_Y','R_ELBOW_P','R_ELBOW_Y','R_WRIST_R','R_WRIST_Y'] +
        ['L_SHOULDER_P','L_SHOULDER_R','L_SHOULDER_Y','L_ELBOW_P','L_ELBOW_Y','L_WRIST_R','L_WRIST_Y']
    ):
        node = mjcf_model.find('joint', joint)
        if node is not None:
            node.remove()
    mjcf_model.equality.remove()

    # ------------------------------------------------------------------
    # Фиксируем конфигурацию плеч/локтей
    # ------------------------------------------------------------------
    for body_name, euler in ARM_BODIES.items():
        bd = mjcf_model.find('body', body_name)
        if bd is not None:
            bd.euler = euler

    # ------------------------------------------------------------------
    # Удаляем лишние collision-geom по классу
    # ------------------------------------------------------------------
    ALLOWED_COLL = {
        'R_HIP_R_S','R_HIP_Y_S','R_KNEE_S',
        'L_HIP_R_S','L_HIP_Y_S','L_KNEE_S'
    }
    for body in mjcf_model.worldbody.find_all('body'):
        for geom in list(body.geom):
            geom.dclass = getattr(geom.dclass, 'dclass', None)
            if geom.dclass == 'collision' and body.name not in ALLOWED_COLL:
                geom.remove()

    # ------------------------------------------------------------------
    # Переносим default class="collision" в группу 3
    # ------------------------------------------------------------------
    for default in mjcf_model.find_all('default'):
        if getattr(default, 'class_', None) == 'collision':
            # default.geom — всегда хотя бы один <geom>
            for geom in default.geom:
                geom.group = '3'

    # ------------------------------------------------------------------
    # Добавляем «статичные» коллизии под щиколотки
    # ------------------------------------------------------------------
    for foot in ['R_ANKLE_P_S','L_ANKLE_P_S']:
        bd = mjcf_model.worldbody.find('body', foot)
        if bd is not None:
            bd.add('geom',
                   dclass='collision',
                   size='0.1 0.05 0.01',
                   pos='0.029 0  -0.09778',
                   type='box')

    # ------------------------------------------------------------------
    # Вновь разрешаем коллизии между некоторыми телами
    # ------------------------------------------------------------------
    mjcf_model.contact.add('exclude', body1='R_KNEE_S', body2='R_ANKLE_P_S')
    mjcf_model.contact.add('exclude', body1='L_KNEE_S', body2='L_ANKLE_P_S')

    # ------------------------------------------------------------------
    # Удаляем неиспользуемые меши
    # ------------------------------------------------------------------
    used_meshes = {g.mesh.name for g in mjcf_model.find_all('geom')
                   if getattr(g, 'type', None) == 'mesh'}
    for mesh in list(mjcf_model.find_all('mesh')):
        if mesh.name not in used_meshes:
            mesh.remove()

    # ------------------------------------------------------------------
    # Правим позиции сенсорных сайтов (если они есть)
    # ------------------------------------------------------------------
    for site_name in ('rf_force','lf_force'):
        st = mjcf_model.find('site', site_name)
        if st is not None:
            st.pos = '0.03 0.0 -0.1'

    # ------------------------------------------------------------------
    # Опционально — вставка ящиков
    # ------------------------------------------------------------------
    if config.get('boxes', False):
        for idx in range(20):
            name = f'box{idx+1:02d}'
            bd = mjcf_model.worldbody.add('body', name=name, pos=[0,0,-0.2])
            bd.add('geom',
                   name=name,
                   dclass='collision',
                   group='0',
                   size='1 1 0.1',
                   type='box')

    # ------------------------------------------------------------------
    # Обёртываем пол в body «floor»
    # ------------------------------------------------------------------
    fl = mjcf_model.find('geom', 'floor')
    if fl is not None:
        fl.remove()
    floor_bd = mjcf_model.worldbody.add('body', name='floor')
    floor_bd.add('geom',
                 name='floor',
                 type='plane',
                 size='0 0 0.25',
                 material='groundplane')

    # ------------------------------------------------------------------
    # Экспортим
    # ------------------------------------------------------------------
    mjcf.export_with_assets(mjcf_model, out_dir=export_path, precision=5)
    out_xml = os.path.join(export_path, mjcf_model.model + '.xml')
    print("Exporting XML model to", out_xml)

if __name__ == '__main__':
    builder(sys.argv[1], config={})
