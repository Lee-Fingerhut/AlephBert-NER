import React, { Component } from 'react';
import styled from 'styled-components';
import './App.css';

class App extends Component {
    state = {
        data: []
    };
    //getting string from textJSON and sending to the look_and_see Function to process the input
    componentDidMount() {
        //storing current path into the variable and use it later to access textJSON.json file
        const path = ${__dirname}
        axios.get('http://127.0.0.1:3000/{path}/textJSON.json')
            .then(res => {
                this.setState({ data: res.data });
            });
    }


    render() {
        const { data } = this.state;
        //mapping each word to index
        const result = data.map((entry, index) => {
            console.log(entry);
            
                    return  <li key={index}>{entry}</li>
                    
        });
        //returns list of words
       return <ul>{result}</ul>;
    }
}
export default App;