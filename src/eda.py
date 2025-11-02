import matplotlib.pyplot as plt

def run_eda(data):
    print(f"Total data points: {len(data)}")
    print(f"Start Date: {data.index.min()}")
    print(f"End Date: {data.index.max()}")
    print(f"Missing Values: {data.isnull().sum().sum()}")

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Sales'], marker='o')
    plt.title("📊 Monthly Sales Overview")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.show()
