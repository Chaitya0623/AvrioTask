import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import altair as alt
import base64

st.title('2D Data Labeling Tool')
st.write("We have used Streamlit which serves as a frontent to create good visualizations for our tool. The entire code could be written in one python file (the one with streamlit), but we have integrated flask for creating a web framework linked via API's.")
st.write('We have deployed the Streamlit app on Streamlit Cloud, and Backend (Flask) on Render.')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    files = {'file': uploaded_file.getvalue()}
    response = requests.post("https://avriotask.onrender.com/upload", files=files)
    if response.status_code == 200:
        st.success("File uploaded successfully.")
        df = pd.DataFrame(response.json())
        st.write('We have used Pandas for reading and writing CSV files.')
        st.write("Data Preview:", df.head())

        st.write('As we just have one dependent variable (clustering based on timestamp will not be that effective, we use Kmeans over DBscan.')
        data = df.drop(columns=['Time'])

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_scaled)
            sse.append(kmeans.inertia_)

        elbow_data = pd.DataFrame({
            'Number of Clusters': range(1, 11),
            'SSE': sse
        })

        elbow_chart = alt.Chart(elbow_data).mark_line(point=True).encode(
            x='Number of Clusters',
            y='SSE'
        ).properties(
            title='Elbow Method'
        )

        st.altair_chart(elbow_chart, use_container_width=True)
        st.write('As we see a sharp edge when k = 2, we use number of clusters = 2.')

        response = requests.post("https://avriotask.onrender.com/cluster", json={"n_clusters": 2})
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            fig = px.scatter(df, x='L1', y='L2', color='label', title="Clustered Data")
            st.write('We have used PCA to reduce the dimensions from 3 to 2.')
            st.plotly_chart(fig)
            st.write('Plotly for creating the scatter plot visualizations, for implementing the rectangular selector and manual label editing GUI. You can select a rectangular box in the following graph and enter the label you want in the input below it.')

            fig.update_layout(dragmode='select')
            selected_points = plotly_events(fig, key="plot", select_event=True)
            new_label = st.text_input("New Label")

            if st.button("Edit Labels"):
                if selected_points is not None:
                    indices = [point['pointIndex'] for point in selected_points]
                    df.loc[indices, 'label'] = new_label
                    fig = px.scatter(df, x='L1', y='L2', color='label', title="Updated Clustered Data")
                    st.plotly_chart(fig)
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to bytes and then to base64
                href = f'<a href="data:file/csv;base64,{b64}" download="labeled_data.csv">Download CSV File</a>'
                st.write("You can download the updated CSV file using the link below.")
                st.markdown(href, unsafe_allow_html=True)

st.write('To access the source code of this tool, use the link below.')
st.markdown("[GitHub Repository](https://github.com/Chaitya0623/AvrioTask)", unsafe_allow_html=True)