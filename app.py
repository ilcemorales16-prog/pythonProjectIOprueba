import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Calculadora IO", layout="wide", page_icon="🧮")

st.title("🧮 Calculadora de Investigación de Operaciones")

# Menú lateral para navegar entre módulos
modulo = st.sidebar.radio(
    "Selecciona un método:",
    ["Método Gráfico", "Método Simplex", "Métodos de Transporte"]
)

# ============================================================
# 1. MÉTODO GRÁFICO
# ============================================================
if modulo == "Método Gráfico":
    st.header("📈 Método Gráfico (Maximización)")
    st.markdown("Resuelve problemas de Programación Lineal con 2 variables de decisión ($X_1$ y $X_2$).")

    col1, col2 = st.columns(2)
    with col1:
        c1 = st.number_input("Coeficiente de $X_1$ en Z:", value=3.0)
    with col2:
        c2 = st.number_input("Coeficiente de $X_2$ en Z:", value=5.0)

    n = st.number_input("Cantidad de restricciones:", min_value=1, max_value=10, value=2, step=1)

    restricciones = []
    st.subheader("Restricciones:")
    for i in range(int(n)):
        col_a, col_b, col_signo, col_d = st.columns([2, 2, 2, 2])
        with col_a:
            a = st.number_input(f"Coeff $X_1$ (R{i + 1})", value=1.0, key=f"g_a_{i}")
        with col_b:
            b = st.number_input(f"Coeff $X_2$ (R{i + 1})", value=1.0, key=f"g_b_{i}")
        with col_signo:
            signo = st.selectbox(f"Signo (R{i + 1})", ["<=", ">=", "="], key=f"g_s_{i}")
        with col_d:
            d = st.number_input(f"Resultado (R{i + 1})", value=4.0, key=f"g_d_{i}")
        restricciones.append([a, b, signo, d])

    if st.button("🚀 Resolver Método Gráfico"):
        restricciones_usuario = list(restricciones)
        restricciones_completas = list(restricciones)
        restricciones_completas.append([1, 0, ">=", 0])
        restricciones_completas.append([0, 1, ">=", 0])

        puntos = []
        for i in range(len(restricciones_completas)):
            for j in range(i + 1, len(restricciones_completas)):
                a1, b1, _, d1 = restricciones_completas[i]
                a2, b2, _, d2 = restricciones_completas[j]
                det = a1 * b2 - a2 * b1
                if det != 0:
                    x1 = (d1 * b2 - d2 * b1) / det
                    x2 = (a1 * d2 - a2 * d1) / det
                    puntos.append([x1, x2])

        puntos_factibles = []
        for p in puntos:
            x1, x2 = p[0], p[1]
            if x1 < -0.000001 or x2 < -0.000001:
                continue
            es_factible = True
            for r in restricciones_completas:
                a, b, signo, d = r
                izquierda = a * x1 + b * x2
                if signo == "<=" and izquierda > d + 0.000001:
                    es_factible = False
                elif signo == ">=" and izquierda < d - 0.000001:
                    es_factible = False
                elif signo == "=" and abs(izquierda - d) > 0.000001:
                    es_factible = False
            if es_factible:
                puntos_factibles.append([x1, x2])

        puntos_unicos = []
        for p in puntos_factibles:
            if not any(abs(p[0] - u[0]) < 1e-5 and abs(p[1] - u[1]) < 1e-5 for u in puntos_unicos):
                puntos_unicos.append(p)

        if puntos_unicos:
            mejor_z = None
            mejor_punto = None
            puntos_tabla = []

            for p in puntos_unicos:
                z = c1 * p[0] + c2 * p[1]
                puntos_tabla.append({"X1": round(p[0], 4), "X2": round(p[1], 4), "Z": round(z, 4)})
                if mejor_z is None or z > mejor_z:
                    mejor_z = z
                    mejor_punto = p

            st.success("### 🎯 Solución Óptima")
            st.metric("Valor Óptimo de Z", round(mejor_z, 4))
            st.write(f"**Punto Óptimo:** $X_1 = {round(mejor_punto[0], 4)}$, $X_2 = {round(mejor_punto[1], 4)}$")

            st.subheader("Vértices Factibles Evaluados")
            st.dataframe(pd.DataFrame(puntos_tabla))

            # Gráfica
            fig, ax = plt.subplots(figsize=(8, 6))
            max_x1 = max([p[0] for p in puntos_unicos] + [1]) * 1.3
            max_x2 = max([p[1] for p in puntos_unicos] + [1]) * 1.3
            x1_vals = np.linspace(0, max_x1, 400)

            for i, r in enumerate(restricciones_usuario):
                a, b, signo, d = r
                if b != 0:
                    x2_vals = (d - a * x1_vals) / b
                    ax.plot(x1_vals, x2_vals, label=f'R{i + 1}: {a}X1 + {b}X2 {signo} {d}')
                else:
                    ax.axvline(x=d / a, label=f'R{i + 1}: {a}X1 {signo} {d}', linestyle='--')

            if len(puntos_unicos) >= 3:
                pts = np.array(puntos_unicos)
                centroide = np.mean(pts, axis=0)
                angulos = np.arctan2(pts[:, 1] - centroide[1], pts[:, 0] - centroide[0])
                pts = pts[np.argsort(angulos)]
                ax.fill(pts[:, 0], pts[:, 1], color='lightgreen', alpha=0.5, label='Región Factible')

            pf_x1 = [p[0] for p in puntos_unicos]
            pf_x2 = [p[1] for p in puntos_unicos]
            ax.scatter(pf_x1, pf_x2, color='blue', zorder=5, label='Vértices Factibles')
            ax.scatter(mejor_punto[0], mejor_punto[1], color='red', s=120, zorder=6,
                       label=f'Óptimo ({round(mejor_punto[0], 2)}, {round(mejor_punto[1], 2)})')

            ax.set_xlim(0, max_x1)
            ax.set_ylim(0, max_x2)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title('Método Gráfico - Programación Lineal')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.axhline(0, color='black', linewidth=1.2)
            ax.axvline(0, color='black', linewidth=1.2)
            ax.legend(loc='upper right')

            st.pyplot(fig)
        else:
            st.error("No existe una solución factible.")

# ============================================================
# 2. MÉTODO SIMPLEX
# ============================================================
elif modulo == "Método Simplex":
    st.header("📊 Método Simplex (Maximización)")
    st.markdown("Para problemas con restricciones de tipo $\leq$ y variables $\geq 0$.")

    col_v, col_r = st.columns(2)
    with col_v:
        num_vars = st.number_input("Número de variables:", min_value=1, max_value=10, value=2, step=1)
    with col_r:
        num_rest = st.number_input("Número de restricciones:", min_value=1, max_value=10, value=2, step=1)

    st.subheader("Función Objetivo (Coeficientes en Z):")
    cols_obj = st.columns(int(num_vars))
    c_obj = []
    for j in range(int(num_vars)):
        with cols_obj[j]:
            val = st.number_input(f"$X_{j+1}$", value=3.0 if j == 0 else 5.0, key=f"s_obj_{j}")
            c_obj.append(val)

    st.subheader("Restricciones:")
    matriz_rest = []
    rhs = []
    for i in range(int(num_rest)):
        st.write(f"**Restricción {i+1}:**")
        cols_r = st.columns(int(num_vars) + 1)
        fila = []
        for j in range(int(num_vars)):
            with cols_r[j]:
                v = st.number_input(f"Coeff $X_{j+1}$", value=1.0, key=f"s_r_{i}_{j}")
                fila.append(v)
        with cols_r[-1]:
            res = st.number_input("Resultado ($\leq$)", value=4.0, key=f"s_rhs_{i}")
            rhs.append(res)
        matriz_rest.append(fila)

    if st.button("🚀 Resolver Método Simplex"):
        v_count = int(num_vars)
        r_count = int(num_rest)

        tabla = []
        for i in range(r_count):
            fila = matriz_rest[i].copy()
            holguras = [1.0 if i == k else 0.0 for k in range(r_count)]
            fila.extend(holguras)
            fila.append(rhs[i])
            tabla.append(fila)

        fila_z = [-val for val in c_obj] + [0.0] * (r_count + 1)
        tabla.append(fila_z)

        encabezados = [f"X{j+1}" for j in range(v_count)] + [f"S{i+1}" for i in range(r_count)] + ["RHS"]

        st.subheader("Iteraciones de la Tabla Simplex")
        contador = 1
        es_ilimitado = False

        while True:
            df_tabla = pd.DataFrame(tabla, columns=encabezados)
            st.write(f"**Tabla Iteración {contador}:**")
            st.dataframe(df_tabla.style.format("{:.2f}"))

            fila_z_actual = tabla[-1]
            menor = min(fila_z_actual[:-1])
            if menor >= 0:
                break

            col_pivote = fila_z_actual.index(menor)

            menor_ratio = None
            fila_pivote = -1
            for i in range(r_count):
                elem = tabla[i][col_pivote]
                if elem > 0:
                    ratio = tabla[i][-1] / elem
                    if menor_ratio is None or ratio < menor_ratio:
                        menor_ratio = ratio
                        fila_pivote = i

            if fila_pivote == -1:
                st.error("El problema es ilimitado.")
                es_ilimitado = True
                break

            pivote = tabla[fila_pivote][col_pivote]
            tot_cols = len(encabezados)

            for j in range(tot_cols):
                tabla[fila_pivote][j] /= pivote

            for i in range(len(tabla)):
                if i != fila_pivote:
                    factor = tabla[i][col_pivote]
                    for j in range(tot_cols):
                        tabla[i][j] -= factor * tabla[fila_pivote][j]

            contador += 1

        if not es_ilimitado:
            st.success("### 🎯 Solución Óptima Encontrada")
            solucion = []
            for j in range(v_count):
                val = 0.0
                fila_var = -1
                es_basica = True
                unos = 0
                for i in range(r_count):
                    if abs(tabla[i][j] - 1.0) < 1e-5:
                        unos += 1
                        fila_var = i
                    elif abs(tabla[i][j]) > 1e-5:
                        es_basica = False
                if es_basica and unos == 1:
                    val = tabla[fila_var][-1]
                solucion.append(val)

            cols_res = st.columns(v_count + 1)
            for j in range(v_count):
                with cols_res[j]:
                    st.metric(f"X{j+1}", round(solucion[j], 4))
            with cols_res[-1]:
                st.metric("Z (Máximo)", round(tabla[-1][-1], 4))

# ============================================================
# 3. MÉTODOS DE TRANSPORTE
# ============================================================
elif modulo == "Métodos de Transporte":
    st.header("🚚 Métodos de Transporte")

    metodo_t = st.selectbox("Selecciona el algoritmo:", ["Esquina Noroeste", "Costo Mínimo", "Aproximación de Vogel"])

    col_o, col_d = st.columns(2)
    with col_o:
        n_orig = st.number_input("Cantidad de Orígenes:", min_value=1, max_value=10, value=3, step=1)
    with col_d:
        n_dest = st.number_input("Cantidad de Destinos:", min_value=1, max_value=10, value=4, step=1)

    st.subheader("Matriz de Costos:")
    costos = []
    for i in range(int(n_orig)):
        cols_c = st.columns(int(n_dest))
        fila_c = []
        for j in range(int(n_dest)):
            with cols_c[j]:
                v = st.number_input(f"O{i+1} $\\rightarrow$ D{j+1}", value=float((i+1)*(j+1)+2), key=f"t_c_{i}_{j}")
                fila_c.append(v)
        costos.append(fila_c)

    col_oferta, col_demanda = st.columns(2)
    with col_oferta:
        st.subheader("Oferta por Origen:")
        oferta = []
        for i in range(int(n_orig)):
            val_o = st.number_input(f"Oferta O{i+1}:", value=100.0, key=f"t_o_{i}")
            oferta.append(val_o)

    with col_demanda:
        st.subheader("Demanda por Destino:")
        demanda = []
        for j in range(int(n_dest)):
            val_d = st.number_input(f"Demanda D{j+1}:", value=75.0, key=f"t_d_{j}")
            demanda.append(val_d)

    if st.button("🚀 Resolver Transporte"):
        if sum(oferta) != sum(demanda):
            st.warning(f"⚠️ El problema no está balanceado. Oferta Total: {sum(oferta)} | Demanda Total: {sum(demanda)}")

        filas = int(n_orig)
        cols = int(n_dest)
        asignaciones = np.zeros((filas, cols))
        oferta_t = oferta.copy()
        demanda_t = demanda.copy()

        if metodo_t == "Esquina Noroeste":
            i, j = 0, 0
            while i < filas and j < cols:
                cant = min(oferta_t[i], demanda_t[j])
                asignaciones[i][j] = cant
                oferta_t[i] -= cant
                demanda_t[j] -= cant
                if oferta_t[i] == 0:
                    i += 1
                if demanda_t[j] == 0:
                    j += 1

        elif metodo_t == "Costo Mínimo":
            while True:
                menor_costo = None
                m_i, m_j = -1, -1
                for i in range(filas):
                    for j in range(cols):
                        if oferta_t[i] > 0 and demanda_t[j] > 0:
                            if menor_costo is None or costos[i][j] < menor_costo:
                                menor_costo = costos[i][j]
                                m_i, m_j = i, j
                if m_i == -1:
                    break
                cant = min(oferta_t[m_i], demanda_t[m_j])
                asignaciones[m_i][m_j] = cant
                oferta_t[m_i] -= cant
                demanda_t[m_j] -= cant

        elif metodo_t == "Aproximación de Vogel":
            f_act = [True] * filas
            c_act = [True] * cols
            while sum(oferta_t) > 0 and sum(demanda_t) > 0:
                p_filas = []
                for i in range(filas):
                    vals = [costos[i][j] for j in range(cols) if f_act[i] and c_act[j]]
                    if len(vals) >= 2:
                        vals.sort()
                        p_filas.append(vals[1] - vals[0])
                    elif len(vals) == 1:
                        p_filas.append(vals[0])
                    else:
                        p_filas.append(-1)

                p_cols = []
                for j in range(cols):
                    vals = [costos[i][j] for i in range(filas) if f_act[i] and c_act[j]]
                    if len(vals) >= 2:
                        vals.sort()
                        p_cols.append(vals[1] - vals[0])
                    elif len(vals) == 1:
                        p_cols.append(vals[0])
                    else:
                        p_cols.append(-1)

                max_f = max(p_filas)
                max_c = max(p_cols)

                if max_f >= max_c:
                    i = p_filas.index(max_f)
                    m_j = -1
                    m_cost = None
                    for j in range(cols):
                        if c_act[j]:
                            if m_cost is None or costos[i][j] < m_cost:
                                m_cost = costos[i][j]
                                m_j = j
                    j = m_j
                else:
                    j = p_cols.index(max_c)
                    m_i = -1
                    m_cost = None
                    for i in range(filas):
                        if f_act[i]:
                            if m_cost is None or costos[i][j] < m_cost:
                                m_cost = costos[i][j]
                                m_i = i
                    i = m_i

                cant = min(oferta_t[i], demanda_t[j])
                asignaciones[i][j] = cant
                oferta_t[i] -= cant
                demanda_t[j] -= cant

                if oferta_t[i] == 0:
                    f_act[i] = False
                if demanda_t[j] == 0:
                    c_act[j] = False

        st.success(f"### 🎯 Asignación Concluida - {metodo_t}")
        df_asig = pd.DataFrame(
            asignaciones,
            index=[f"Origen {i+1}" for i in range(filas)],
            columns=[f"Destino {j+1}" for j in range(cols)]
        )
        st.dataframe(df_asig)

        costo_total = sum(asignaciones[i][j] * costos[i][j] for i in range(filas) for j in range(cols))
        st.metric("💰 Costo Total de Transporte", f"${costo_total:,.2f}")