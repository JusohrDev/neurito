"""Experimentos de la GUIA_LABERINTO.md ejecutados sin Pygame.

Uso:  uv run python experimentos/experimentos_laberinto.py
"""
import heapq
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from laberinto_mundo import (MAPAS, Laberinto, libres, vecino, valido,
                             distancias_bfs, accion_experta, elegir_accion)
import laberinto_red as lr
from laberinto_red import RedLaberinto, codificar

LIMITE = 45          # mismo límite de intentos que el juego
EPISODIOS = 300      # episodios por evaluación (100 por mapa)


# ---------------------------------------------------------------- simulación
def jugar(red, asistido, episodios=EPISODIOS, semilla=7):
    """Devuelve % comidas, pasos medios en éxitos, choques/ep y exceso sobre ruta óptima."""
    mundo = Laberinto(semilla=semilla)
    exitos, pasos, choques, exceso = 0, [], 0, []
    for e in range(episodios):
        mundo.reiniciar(e % len(MAPAS))
        optimo = distancias_bfs(mundo.mapa, mundo.meta)[mundo.posicion]
        antes = mundo.choques
        while True:
            a, _ = elegir_accion(red, mundo, mundo.visitadas, asistido)
            r = mundo.mover(a)
            if r == 'comida':
                exitos += 1
                pasos.append(mundo.pasos)
                exceso.append(mundo.pasos - optimo)
                break
            if mundo.intentos >= LIMITE:
                break
        choques += mundo.choques - antes
    return {
        'comidas_%': round(100 * exitos / episodios, 1),
        'pasos_medios': round(float(np.mean(pasos)), 2) if pasos else None,
        'pasos_extra_vs_optimo': round(float(np.mean(exceso)), 2) if exceso else None,
        'choques_por_ep': round(choques / episodios, 2),
    }


def entrenar(red, X, y, epocas):
    for _ in range(epocas):
        red.entrenar_epoca(X, y)


def resumen(red, X, y, Xt, yt):
    return {
        'epocas': red.epocas,
        'perdida': round(red.historial[-1], 4) if red.historial else None,
        'exactitud_entreno_%': round(100 * red.exactitud(X, y), 1),
        'exactitud_prueba_%': round(100 * red.exactitud(Xt, yt), 1),
        'red_pura': jugar(red, False),
        'asistido': jugar(red, True),
    }


# ------------------------------------------------- expertos alternativos (P7)
def primer_paso(camino):
    (f0, c0), (f1, c1) = camino[0], camino[1]
    return lr_dir.index((f1 - f0, c1 - c0))


lr_dir = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def experto_bfs(mapa, origen, meta):
    return accion_experta(mapa, origen, distancias_bfs(mapa, meta))


def experto_dfs(mapa, origen, meta):
    """DFS iterativo: primer camino que encuentra (no necesariamente el más corto)."""
    pila, padre = [origen], {origen: None}
    while pila:
        actual = pila.pop()
        if actual == meta:
            break
        for a in reversed(range(4)):   # explora ARRIBA primero
            s = vecino(actual, a)
            if valido(mapa, s) and s not in padre:
                padre[s] = actual
                pila.append(s)
    camino, n = [], meta
    while n is not None:
        camino.append(n)
        n = padre[n]
    return primer_paso(camino[::-1])


def experto_astar(mapa, origen, meta):
    """A* con heurística Manhattan (admisible: siempre da ruta óptima)."""
    h = lambda p: abs(p[0] - meta[0]) + abs(p[1] - meta[1])
    abierto, g, padre, cont = [(h(origen), 0, origen)], {origen: 0}, {origen: None}, 0
    expandidos = 0
    while abierto:
        _, _, actual = heapq.heappop(abierto)
        expandidos += 1
        if actual == meta:
            break
        for a in range(4):
            s = vecino(actual, a)
            if valido(mapa, s) and g[actual] + 1 < g.get(s, 10**9):
                g[s] = g[actual] + 1
                padre[s] = actual
                cont += 1
                heapq.heappush(abierto, (g[s] + h(s), cont, s))
    camino, n = [], meta
    while n is not None:
        camino.append(n)
        n = padre[n]
    return primer_paso(camino[::-1])


def generar_datos_con(experto, semilla=42, fraccion_prueba=0.2):
    muestras = []
    for i, mapa in enumerate(MAPAS):
        celdas = libres(mapa)
        for meta in celdas:
            for pos in celdas:
                if pos != meta:
                    muestras.append((i, pos, meta, experto(mapa, pos, meta)))
    rng = np.random.default_rng(semilla)
    rng.shuffle(muestras)
    corte = int(len(muestras) * (1 - fraccion_prueba))
    def arr(filas):
        return (np.stack([codificar(i, p, m) for i, p, m, _ in filas]),
                np.asarray([a for *_, a in filas], dtype=np.int64))
    return arr(muestras[:corte]), arr(muestras[corte:])


def experto_es_optimo(experto):
    """% de etiquetas que coinciden con algún movimiento de ruta más corta."""
    ok = tot = 0
    for mapa in MAPAS:
        celdas = libres(mapa)
        for meta in celdas:
            d = distancias_bfs(mapa, meta)
            for pos in celdas:
                if pos == meta:
                    continue
                a = experto(mapa, pos, meta)
                ok += d[vecino(pos, a)] == d[pos] - 1
                tot += 1
    return round(100 * ok / tot, 1)


def ruta_del_experto(experto):
    """Sigue al experto sin red: ¿llega? ¿cuántos pasos extra?"""
    extra, fallos = [], 0
    for mapa in MAPAS:
        celdas = libres(mapa)
        for meta in celdas:
            d = distancias_bfs(mapa, meta)
            for pos in celdas:
                if pos == meta:
                    continue
                p, n = pos, 0
                while p != meta and n < 200:
                    p = vecino(p, experto(mapa, p, meta))
                    n += 1
                if p == meta:
                    extra.append(n - d[pos])
                else:
                    fallos += 1
    return {'llega_%': round(100 * len(extra) / (len(extra) + fallos), 1),
            'pasos_extra_medios': round(float(np.mean(extra)), 2)}


# ---------------------------------------------------------------- principal
def main():
    res = {}
    (X, y), (Xt, yt), _ = lr.generar_datos()
    res['datos'] = {'entreno': len(X), 'prueba': len(Xt)}

    # P1 sin entrenar
    red = RedLaberinto()
    res['p1_sin_entrenar'] = resumen(red, X, y, Xt, yt)

    # P2 100 / 200 / 400 épocas
    res['p2_epocas'] = []
    red = RedLaberinto()
    for objetivo in (100, 200, 400):
        entrenar(red, X, y, objetivo - red.epocas)
        res['p2_epocas'].append(resumen(red, X, y, Xt, yt))
        print('épocas', objetivo, 'ok')

    # P4 neuronas ocultas (200 épocas)
    res['p4_ocultas'] = []
    for n in (16, 32, 96, 128):
        red = RedLaberinto(ocultas=n)
        entrenar(red, X, y, 200)
        r = resumen(red, X, y, Xt, yt)
        r['ocultas'] = n
        res['p4_ocultas'].append(r)
        print('ocultas', n, 'ok')

    # P5 tasa de aprendizaje (200 épocas)
    res['p5_tasa'] = []
    for t in (0.01, 0.1, 0.45, 1.0, 2.0):
        red = RedLaberinto(tasa=t)
        entrenar(red, X, y, 200)
        r = resumen(red, X, y, Xt, yt)
        r['tasa'] = t
        h = red.historial
        r['perdida_ep10'] = round(h[9], 4)
        r['oscilacion_ult20'] = round(float(np.std(np.diff(h[-20:]))), 5)
        r['finita'] = bool(np.isfinite(h[-1]))
        res['p5_tasa'].append(r)
        print('tasa', t, 'ok')

    # P6 por mapa, con la red de 400 épocas, sólo pares de prueba
    red = RedLaberinto()
    entrenar(red, X, y, 400)
    _, _, filas_prueba = lr.generar_datos()
    por_mapa = []
    for i in range(len(MAPAS)):
        ids = [k for k, f in enumerate(filas_prueba) if f[0] == i]
        acc = float(np.mean(np.argmax(red.adelante(Xt[ids])[2], 1) == yt[ids]))
        # episodios cuyo par (inicio, meta) es exactamente de prueba
        exitos = 0
        pares = [(f[1], f[2]) for f in filas_prueba if f[0] == i]
        rnd = random.Random(1)
        muestra = rnd.sample(pares, min(150, len(pares)))
        for pos, meta in muestra:
            m = Laberinto(indice=i)
            m.meta, m.posicion, m.visitadas = meta, pos, {pos: 1}
            while m.intentos < LIMITE:
                a, _ = elegir_accion(red, m, m.visitadas, False)
                if m.mover(a) == 'comida':
                    exitos += 1
                    break
        por_mapa.append({'mapa': i + 1, 'ejemplos_prueba': len(ids),
                         'exactitud_prueba_%': round(100 * acc, 1),
                         'comidas_red_pura_pares_prueba_%': round(100 * exitos / len(muestra), 1)})
    res['p6_mapas'] = por_mapa

    # P7 DFS vs BFS vs A*
    res['p7_expertos'] = []
    for nombre, exp in (('BFS', experto_bfs), ('DFS', experto_dfs), ('A*', experto_astar)):
        (Xe, ye), (Xte, yte) = generar_datos_con(exp)
        red = RedLaberinto()
        entrenar(red, Xe, ye, 400)
        r = resumen(red, Xe, ye, Xte, yte)
        r['experto'] = nombre
        r['etiquetas_optimas_%'] = experto_es_optimo(exp)
        r['experto_solo'] = ruta_del_experto(exp)
        res['p7_expertos'].append(r)
        print('experto', nombre, 'ok')

    salida = Path(__file__).with_name('resultados_laberinto.json')
    salida.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
