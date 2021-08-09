import React, { Component } from 'react';
import styled from 'styled-components';
import './App.css';

class App extends Component {
    state = {
        data: []
    };


    componentDidMount() {
        const url = "https://en.wikipedia.org/w/api.php?action=opensearch&search=Seona+Dancing&format=json&origin=*&limit=1";

        fetch(url)
            .then(result => result.json())
            .then(result => {
                this.setState({
                    data: result
                })
            });
    }


    componentDidMount() {
        axios.get('http://127.0.0.1:3000/E:/NER_Project-main/APP/Client/src/textJSON.json')
            .then(res => {
                this.setState({ data: res.data });
            });
    }


    render() {
        const { data } = this.state;

        const result = data.map((entry, index) => {
            console.log(entry);
            
                    return  <li key={index}>{entry}</li>
                    
        });

       return <ul>{result}</ul>;
    }
}
export default App;