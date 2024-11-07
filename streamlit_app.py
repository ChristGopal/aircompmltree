import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from six import StringIO
import io
from sklearn.tree import export_graphviz
import pydotplus

# Load your data (replace 'air_comp_data_new.csv' with your actual file path)
data = pd.read_csv('air_comp_data_new.csv')

data.columns
data.head()
data[data.isnull().any(axis=1)].head()
clean_data = data.copy()
clean_data['high_flow_label'] = (clean_data['air_flow'] >900.00) *1
clean_data['high_flow_label'].head()
y = clean_data[['high_flow_label']].copy()
clean_data['air_flow'].head()
morning_features = ['rpm', 'motor_power', 'torque', 'outlet_pressure_bar', 'noise_db', 'outlet_temp',
                    'wpump_outlet_press', 'water_inlet_temp', 'water_outlet_temp', 'wpump_power',
                    'water_flow', 'oilpump_power', 'oil_tank_temp', 'gaccx', 'gaccy', 'gaccz',
                    'haccx', 'haccy', 'haccz', 'bearings', 'wpump', 'radiator', 'exvalve'
]

x=clean_data[morning_features].copy()
x.columns
y.columns
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=324)
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=15,random_state=0, criterion='entropy')
humidity_classifier.fit(X_train,y_train)
type(humidity_classifier)
y_predicted = humidity_classifier.predict(X_test)
y_predicted[:10]
y_test['high_flow_label'][:10]
accuracy_score(y_test,y_predicted)*100
print(accuracy_score(y_test,y_predicted)*100)

# Streamlit app
st.title("Decision Tree Classifier")
# Display the decision tree
dot_data = io.StringIO()  # Use io.StringIO instead of StringIO
export_graphviz(humidity_classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Use st.image() to display the image
st.image(graph.create_png(), caption='Decision Tree', use_column_width=True)
# Optionally, display other information like accuracy
st.write(f"Accuracy: {accuracy_score(y_test, y_predicted) * 100:.2f}%")
st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
