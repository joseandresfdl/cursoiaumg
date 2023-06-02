import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv('creditcard.csv')

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['Class', 'Time'], axis=1), df['Class'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

precision = tf.keras.metrics.Precision(name='precision')
recall = tf.keras.metrics.Recall(name='recall')
auc = tf.keras.metrics.AUC(name='auc')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, auc])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    callbacks=[callback]
                   )
test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
print('Test Precision:', test_prec)
print('Test Recall:', test_rec)
print('Test AUC:', test_auc)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 50})
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1score_rf = f1_score(y_test, y_pred_rf, average='weighted')
auc_rf = roc_auc_score(y_test, y_pred_rf)

xgb = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=50)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1score_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
auc_xgb = roc_auc_score(y_test, y_pred_xgb)

print('Random Forest Classifier:')
print('Precision:', precision_rf)
print('Recall:', recall_rf)
print('F1-score:', f1score_rf)
print('AUC:', auc_rf)

print('\nXGBoost Classifier:')
print('Precision:', precision_xgb)
print('Recall:', recall_xgb)
print('F1-score:', f1score_xgb)
print('AUC:', auc_xgb)