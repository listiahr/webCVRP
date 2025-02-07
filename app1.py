import streamlit as st
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

def initialize_population(pop_size, n_cust):
    lower_bound = 1
    upper_bound = n_cust
    population = np.zeros((pop_size, n_cust))
    for i in range(pop_size):
        for j in range(n_cust):
            rand = random.random()
            population[i, j] = lower_bound + rand * (upper_bound - lower_bound)
    return population

def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance

def validate_cvrp_route(route, demands, vehicle_capacity):
    if route[0] != 0 or route[-1] != 0:
        return False, "Rute harus dimulai dari depot dan berakhir di depot"

    visited_customers = set()
    route_demand = 0 
    for i in range(len(route) - 1):
        current_node = route[i]
        next_node = route[i + 1]

        if current_node == next_node:
            return False, "Kendaraan harus meninggalkan titik yang dikunjungi"

        if current_node != 0:
            if current_node in visited_customers:
                return False, f"Customer {current_node} dikunjungi lebih dari satu kali"
            if current_node not in demands:
                return False, f"Customer {current_node} tidak memiliki pesanan"

            route_demand += demands[current_node]
            visited_customers.add(current_node)

        if route_demand > vehicle_capacity:
            return False, f"Rute ini melebihi kapasitas kendaraan dengan jumlah pesanan: {route_demand}."

    total_customers = set(demands.keys()) - {0}
    if visited_customers != total_customers:
        return False, "Tidak semua pelanggan dikunjungi"

    return True

def create_route(route, demands, vehicle_capacity):
    depot = 0
    sorted_indices = np.argsort(route)
    sorted_route = [depot] + [int(index)+1 for index in sorted_indices] + [depot]
    is_valid = validate_cvrp_route(sorted_route, demands, vehicle_capacity)
    if is_valid:
        return sorted_route

def calculate_total_biaya(distance, gaji_supir, biaya_maintain, harga_BBM, rasio_bbm_jarak):
    pengeluaran_bbm = (distance/rasio_bbm_jarak) * harga_BBM
    total_biaya = gaji_supir + biaya_maintain + pengeluaran_bbm
    return total_biaya

def create_population_data(population):
    data_poa = {}
    for i in range(len(population)):     
        route = create_route(population[i], demands, vehicle_capacity)
        distance = calculate_total_distance(route, distance_matrix)
        total_biaya = calculate_total_biaya(distance, gaji_supir, biaya_maintain, harga_BBM, rasio_bbm_jarak)
        data_poa['pelican ' + str(i+1)] = {
            'rute' : route,
            'jarak' : int(distance),
            'biaya' : int(total_biaya)
        }
    return data_poa

def main_POA(pop_size, n_cust, distance_matrix, demands, vehicle_capacity, max_iterations, gaji_supir, biaya_maintain, harga_BBM, rasio_bbm_jarak):

    population = initialize_population(pop_size, n_cust)
    best_pelican = None
    best_iteration = None
    best_distance = float('inf')
    population_data = {}

    for iteration in range(max_iterations):
        new_population = []
        population_distance = []
        target_pelican = population[np.random.choice(population.shape[0])]
        target_route = create_route(target_pelican.tolist(), demands, vehicle_capacity)
        target_distance = calculate_total_distance(target_route, distance_matrix)

        for i in range(len(population)):
            new_position_value_p1 = []
            new_position_value_p2 = []
            pelican_route = create_route(population[i], demands, vehicle_capacity)
            pelican_distance = calculate_total_distance(pelican_route, distance_matrix)

            for j in range(len(population[i])):
                rand_num = np.random.random()
                var_I = np.random.choice([1, 2])
                if target_distance < pelican_distance:
                    new_position = population[i][j] + rand_num * (target_pelican[j] - var_I * population[i][j])
                    new_position_value_p1.append(new_position)
                else:
                    new_position = population[i][j] + rand_num * (population[i][j] - target_pelican[j])
                    new_position_value_p1.append(new_position)

            new_route = create_route(new_position_value_p1, demands, vehicle_capacity)
            new_distance = calculate_total_distance(new_route, distance_matrix)
            new_population.append(new_position_value_p1 if new_distance < pelican_distance else population[i])

            pelican_route_p2 = create_route(new_population[i], demands, vehicle_capacity)
            pelican_distance_p2 = calculate_total_distance(pelican_route_p2, distance_matrix)
            for j in range(len(new_population[i])):
                var_R = 0.2
                rand_num = np.random.random()
                var_t = iteration
                var_T = max_iterations

                new_position_p2 = new_population[i][j] + var_R * (1 - (var_t / var_T)) * (2 * rand_num - 1) * new_population[i][j]
                new_position_value_p2.append(new_position_p2)

            new_route_p2 = create_route(new_position_value_p2, demands, vehicle_capacity)
            new_distance_p2 = calculate_total_distance(new_route_p2, distance_matrix)
            
            new_population[i] = new_position_value_p2 if new_distance_p2 < pelican_distance_p2 else new_population[i]

            current_best_route = create_route(new_population[i], demands, vehicle_capacity)
            current_best_distance = calculate_total_distance(current_best_route, distance_matrix)

            if current_best_distance < best_distance :
                best_iteration = iteration+1
                best_pelican = 'pelican ' + str(i+1)
                best_distance = current_best_distance
                best_route = current_best_route
                biaya_optimum = calculate_total_biaya(best_distance, gaji_supir, biaya_maintain, harga_BBM, rasio_bbm_jarak)

        population = np.array(new_population)
        best_solution = {'iterasi' : best_iteration, 'best_pelican' : best_pelican , 'best_route' : best_route, 'best_distance' : int(best_distance), 'total_biaya' : int(biaya_optimum)}
        population_data[iteration+1] = {
            'iterasi' : iteration+1,
            'hasil' : create_population_data(population),
            'best_distance' : best_solution.get('best_distance')
            }

    return best_solution, population_data

st.set_page_config(page_title="CVRP Optimizer", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "App Instruction", "About App"],
        icons=["house", "info-circle", "person-circle"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f0f5"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "color": "black"},
            "nav-link-selected": {"background-color": "#28a745", "color": "white"},
        },
    )

if selected == "Home":
    st.markdown(
    """
    <h1 style="text-align: center;">CVRP Web Optimizer</h1>
    """,
    unsafe_allow_html=True
    )
    st.divider()

    uploaded_file = st.file_uploader("Upload File Excel (Distance Matrix)", type=["xlsx"])
    distance_matrix = None
    if uploaded_file:
        try:
            distance_matrix = pd.read_excel(uploaded_file, header=None).values
            st.success("File uploaded successfully!")
            st.write("Distance Matrix:")
            st.dataframe(distance_matrix)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            distance_matrix = None

    st.header("Input Parameters")

    vehicle_capacity = st.number_input(
        "Vehicle Capacity (Maximum goods per vehicle)",
        min_value=10,
        value=100,
        max_value=150,
        step=1,
        help="The maximum load capacity of a vehicle."
    )

    n_cust = st.number_input(
        "Number of Customers",
        min_value=2,
        step=1,
        help="The total number of customers in the distance matrix."
    )

    if distance_matrix is not None and n_cust > distance_matrix.shape[1] - 1:
        st.error(f"Number of customers cannot exceed {distance_matrix.shape[1] - 1}")

    st.subheader("Customer Name and Demand")
    customers = []
    total_demand = 0
    for i in range(1, n_cust + 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            customer_name = st.text_input(f"Customer Name {i}", key=f"name_{i}")
        with col2:
            customer_demand = st.number_input(
                f"Demand (Customer {i})",
                min_value=1,
                step=1,
                key=f"demand_{i}",
                help="Demand of the customer."
            )
        if customer_name:
            customers.append({"name": customer_name, "demand": customer_demand})
            total_demand += customer_demand
        if not customer_name:
            st.error(f"Customer name {i} can't be empty.")
            input_error = True        

    if total_demand > vehicle_capacity:
        st.error(f"Total demand ({total_demand}) exceeds vehicle capacity ({vehicle_capacity}).")

    st.subheader("Algorithm Parameters")
    pop_size = st.number_input(
        "Population Size",
        min_value=1,
        value=10,
        step=1,
        help="Number of solutions to be considered in each iteration."
    )
    max_iterations = st.number_input(
        "Maximum Iterations",
        min_value=1,
        value=10,
        step=1,
        help="Number of iterations to optimize the route."
    )
    st.subheader("Total Cost Parameter")
    gaji_supir = st.number_input(
        "Driver Salary (Rp)",
        min_value=1000,
        value=50000,
        step=1000,
        help="Salary paid to the driver per trip."
    )
    biaya_maintain = st.number_input(
        "Vehicle Maintenance Cost (Rp)",
        min_value=1000,
        value=5000,
        step=1000,
        help="Cost of maintaining the vehicle per trip."
    )
    harga_BBM = st.number_input(
        "Fuel Cost/Litre (Rp)",
        min_value=7500,
        value=10000,
        step=100,
        help="Cost of fuel per litre."
    )
    rasio_bbm_jarak = st.number_input(
        "Fuel Consumption per Distance (meter/litre)",
        min_value=1000,
        value=10000,
        step=100,
        help="Fuel consumption rate based on distance."
    )

    if st.button("Calculate"):
        if not uploaded_file:
            st.error("Please upload a distance matrix file.")
        elif total_demand > vehicle_capacity:
            st.error("Correct total demand to not exceed vehicle capacity.")
        elif n_cust > distance_matrix.shape[1] - 1:
            st.error(f"Number of customers exceeds allowed maximum.")
        else:
            st.success("Calculation completed successfully! Displaying results...")
            try:
                demands = {0: 0} 
                for i, customer in enumerate(customers):
                    demands[i + 1] = customer['demand']

                population = initialize_population(pop_size, n_cust)

                best_solution, population_data = main_POA(
                    pop_size, n_cust, distance_matrix, demands, vehicle_capacity,
                    max_iterations, gaji_supir, biaya_maintain, harga_BBM, rasio_bbm_jarak
                )

                st.divider()
                st.subheader("Optimization Result")
                st.write("Best Iteration:", best_solution["iterasi"])
                st.write(f"Best Pelican: {best_solution["best_pelican"]}")
                best_route_names = ["Depot" if i == 0 else customers[i - 1]["name"] for i in best_solution["best_route"]]
                st.write("Best Route:", " -> ".join(best_route_names))
                st.write(f"Total Distance: {best_solution["best_distance"]} m")
                st.write(f"Total Cost: Rp {best_solution["total_biaya"]:,}")
                
                st.divider()
                st.subheader("Total Distance each Iteration Graph")
                iterasi = []
                best_distances = []
                for key, value in population_data.items():
                    iterasi.append(value['iterasi'])
                    best_distances.append(value['best_distance'])

                plt.figure(figsize=(6, 4))
                plt.plot(iterasi, best_distances, marker='o', linestyle='-')
                plt.title('Total Distance each Iteration')
                plt.xlabel('Iteration')
                plt.ylabel('Distance')
                plt.grid(True)
                plt.xticks(ticks=range(1, max(iterasi) + 1, 1))
                st.pyplot(plt)

                st.divider()
                st.subheader("Result of each Iteration")
                for iteration, data in population_data.items():
                    with st.expander(f"iteration {iteration}"):
                        st.subheader(f"**Iteration {iteration}:**")
                        st.write(f"Best Distance: {data['best_distance']}")
                        st.divider()
                        for pelican, pelican_data in data['hasil'].items():
                            st.subheader(f"- {pelican}:")
                            best_route_names = ["Depot" if i == 0 else customers[i - 1]["name"] for i in pelican_data["rute"]]
                            st.write(f"    Route: {' -> '.join(map(str, best_route_names))}")
                            st.write(f"    Distance: {pelican_data['jarak']} m")
                            st.write(f"    Cost: Rp {pelican_data['biaya']:,}")
                            st.divider()

            except Exception as e:
                st.error(f"Error: {e}")

if selected == "App Instruction":
    st.markdown(
    """
    <h1 style="text-align: center;">How To Use This Application</h1>
    """,
    unsafe_allow_html=True
    )
    st.divider()
    st.markdown("""
    ### Steps:
    1. **Upload Excel File:**
        - The file must contain a distance matrix between customers in a table format. 
        - Accepted file format: `.xlsx`.
        - The distance matrix cannot be empty.
        - Distance matrix input rules: [Click here](https://binusianorg-my.sharepoint.com/personal/listiarini_hadirianti_binus_ac_id/_layouts/15/guestaccess.aspx?share=ETAFLO0CqvBMtuSRvWdMX2kBiEtscvmSlyozA6yKL0zt7A&e=WuSAI0).

    2. **Fill in Parameters:**
        - Enter the calculation parameters with the following conditions:
            1. **Number of Customers**: Must be greater than 1 and not exceed `(number of columns in the distance matrix - 1)`.
            2. **Vehicle Capacity**: The maximum amount of goods the vehicle can carry.
            3. **Customer Name and Demand**: Minimum demand for each customer is 1, and the total demand for a single route must not exceed the vehicle capacity.
            4. **Population Size** and **Maximum Iterations**: These are calculation parameters required for the POA method. Larger population sizes and iterations are expected to yield better results.
                - Maximum population size: 100.
                - Maximum iterations: 100.
            5. **Driver Salary**, **Vehicle Maintenance Cost**, **Fuel Price**, and **Fuel Consumption Ratio** with distance must not be empty to calculate the total distribution cost.

    3. **Calculate Route:**
        - Click the **Calculate Route** button to perform route calculation using the POA method.

    4. **Results:**
        - The system will display the route calculation results based on the input parameters.
        - Displayed results include:
            - Best solution,
            - Best solution graph for each iteration, and
            - Detailed calculation results for each iteration.
    """)


if selected == "About App":
    st.markdown(
    """
    <h1 style="text-align: center;">About Application</h1>
    """,
    unsafe_allow_html=True
    )
    st.divider()
    st.image("asset/informasi-penulis-dosbing.png", use_container_width=True)

    st.divider()
    st.markdown(
    """
    <h1 style="text-align: center;">Calculation Method</h1>
    <br>
    <h3>Pelican Optimization Algorithm</h3>
    The Pelican Optimization Algorithm (POA) was introduced in 2022 by Pavel Trojovsky and Mohammad Dehghani 
    (Trojovsky & Dehghani, 2022). POA is inspired by the behavior and strategies of pelicans when hunting for food. 
    The performance of POA was also compared to eight well-known optimization algorithms, 
    including Particle Swarm Optimizer (PSO), Teaching-Learning Based Optimization (TLBO), 
    Gray Wolf Optimizer (GWO), Whale Optimization Algorithm (WOA), Marine Predators Algorithm (MPA), 
    Tunicate Swarm Algorithm (TSA), Gravitational Search Algorithm (GSA), and Genetic Algorithm (GA).
    <br>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    Check how POA Work : [Click here](https://binusianorg-my.sharepoint.com/personal/listiarini_hadirianti_binus_ac_id/_layouts/15/guestaccess.aspx?share=ETAFLO0CqvBMtuSRvWdMX2kBiEtscvmSlyozA6yKL0zt7A&e=WuSAI0).
    """)
