import requests
from bs4 import BeautifulSoup

def get_info_organic(name):
    """
    Fetches detailed chemical information for an organic compound from ChemSpider.

    Args:
        name (str): The SMILES string or name of the organic compound.

    Returns:
        dict: A dictionary containing various properties of the compound such as name, formula, CAS number, melting point, and other properties.
    """
    url = f'https://www.chemspider.com/Search.aspx?q={name}'
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Initialize a dictionary to store chemical information
        chemical_dict = {'smiles': name}

        # Extract the title tag to get the name and formula of the compound
        title_tag = soup.title
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            title_parts = title_text.split('|')
            chemical_dict['name'] = title_parts[0].strip()
            chemical_dict['formula'] = title_parts[1].strip()

        # Find the meta tag containing the CAS number
        cas_meta_tag = soup.find("meta", {"name": "keywords"})
        if cas_meta_tag:
            cas_content = cas_meta_tag["content"]
            cas_list = cas_content.split(", ")
            cas_number = cas_list[-1] if len(cas_list) > 1 else None
            chemical_dict['cas_number'] = cas_number

        # Find the div containing predicted data for melting point
        predicted_data2 = soup.find("div", {"id": "epiTab"})
        if predicted_data2:
            data_text = predicted_data2.get_text()
            melting_point = data_text.split("Melting Pt (deg C):")[1].split("(")[0].strip()
            chemical_dict['melting_point'] = float(melting_point)

        # Find the div containing detailed properties
        predicted_data = soup.find("div", {
            "id": "ctl00_ctl00_ContentSection_ContentPlaceHolder1_RecordViewTabDetailsControl_prop_ctl_ACDFormView"})
        if predicted_data:
            properties = predicted_data.find_all("tr")

            # Extract each property and store it in the dictionary
            for prop in properties:
                prop_title = prop.find("td", class_="prop_title")
                prop_value = prop.find("td", class_="prop_value_nowrap")

                if prop_title and prop_value:
                    title = prop_title.get_text(strip=True).rstrip(':')
                    value = prop_value.get_text(strip=True)
                    chemical_dict[title] = value

            return chemical_dict
        else:
            print("Predicted data section not found on the page.")
    else:
        print("Failed to retrieve the webpage.")
