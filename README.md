# Previsão do pH Utilizando Múltiplos Modelos

Este projeto tem como objetivo prever o valor do pH com base em um conjunto de dados utilizando múltiplos modelos de aprendizado de máquina. O foco principal é a construção, treinamento e avaliação de modelos de regressão para previsão do pH, com a possibilidade de adicionar novos modelos e personalizar as configurações.

## Instalação

Para instalar e configurar o projeto, siga as etapas abaixo:

1. **Usando requirements.txt**:
    - Clone este repositório:
    ```bash
    git clone <URL_do_repositório>
    ```
    - Navegue até o diretório do projeto:
    ```bash
    cd <diretório_do_projeto>
    ```
    - Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

2. **Usando o Docker**:
    - Se preferir, use o Docker. O Dockerfile está configurado para instalar as dependências automaticamente.
    - Construa a imagem Docker:
    ```bash
    docker build -t previsao-ph .
    ```
    - Execute o container:
    ```bash
    docker run -v <caminho_para_o_diretorio_com_o_dataset>:/app/Dados previsao-ph
    ```

3. **Coloque o dataset** na pasta do projeto com o nome: `Dados Brutos pHDosado - Completo.csv`.

## Uso

### 1. Script Principal
O script principal está localizado em `main.py`, onde o treinamento e a avaliação do modelo são realizados. 

### 2. Configuração dos Modelos
A configuração dos modelos pode ser ajustada no arquivo `models_config.py`. Neste arquivo, você pode criar novos modelos e modificar os parâmetros dos modelos existentes. O arquivo contém um dicionário com a configuração dos modelos que serão treinados.

### 3. Visualização dos Resultados
Para visualizar os resultados da previsão, como gráficos e métricas de desempenho, use o notebook `visualization.ipynb`. Nele, você pode importar os resultados e explorar as predições geradas pelos modelos.

## Exemplo de Execução

Após a configuração do ambiente e do dataset, basta executar o script principal:

```bash
python main.py
